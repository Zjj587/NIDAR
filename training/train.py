"""Train the public U-Net + DeepLab IRNet; STN stays frozen in data preparation."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "inference"))
from src.models.checkpoint import load_checkpoint
from src.models.deep_intrinsic_net import DeepIntrinsicIntensityNet
from training.data import IntensityDataset, epoch_batches, load_manifest, sha256
from training.losses import intrinsic_loss
from training.historical import RECIPE, DenseObjective


def save_checkpoint(path, checkpoint):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_preview(path, sample, outputs, dense=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r, s, pred, recon = [v[0, 0].detach().cpu().numpy() for v in outputs]
    target = sample["target"][0, 0].numpy()
    mask = sample["mask"][0, 0].numpy()
    images = [sample["rgb"][0].numpy(), sample["input"][0, 0].numpy(), r, s,
              recon, np.ma.masked_where(~mask, target), pred,
              np.ma.masked_where(~mask, np.abs(pred - target))]
    labels = ["RGB", "Pseudo-NIR input", "R (raw)", "S (display only)",
              "R * S", "LiDAR GT (sparse)", "Intensity head", "Absolute error (valid)"]
    if dense:
        labels[5] = "LiDAR-derived dense target"
        labels[7] = "Absolute error (training support)"
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), constrained_layout=True)
    cmap = plt.get_cmap("gray").copy()
    cmap.set_bad("#7a9caa")
    for index, (ax, image, label) in enumerate(zip(axes.flat, images, labels)):
        if index == 0:
            ax.imshow(image)
        else:
            ax.imshow(image, cmap=cmap, vmin=0, vmax=max(float(s.max()), 1e-6) if index == 3 else 1)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_epoch(model, loader, device, loss_config, optimizer=None, recipe="sparse-v1", epoch=0):
    model.train(optimizer is not None)
    sums = {}
    historical = recipe == RECIPE
    objective = DenseObjective(**loss_config, epoch=epoch).to(device) if historical else None
    count = pixels = 0
    batches = 0
    absolute = squared = 0.0
    preview = None
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            x, target, mask = [batch[name].to(device) for name in ("input", "target", "mask")]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            r, s, pred, _ = outputs
            total, terms = (objective(x, r, s, pred, target, mask) if historical else
                            intrinsic_loss(x, r, s, pred, target, mask, **loss_config))
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite loss; checkpoint not updated")
            if optimizer is not None:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 if historical else 10.0,
                                               error_if_nonfinite=True)
                optimizer.step()
            n = x.shape[0]
            count += n
            batches += 1
            for name, value in {**terms, "total": total}.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * (1 if historical else n)
            error = (pred.detach() - target)[mask]
            absolute += float(error.abs().sum())
            squared += float(error.square().sum())
            pixels += error.numel()
            if optimizer is None and preview is None:
                preview = (batch, tuple(v.detach().cpu() for v in outputs))
    if not count or not pixels:
        raise ValueError("Empty epoch or no valid LiDAR pixels")
    metrics = {name: value / (batches if historical else count) for name, value in sums.items()}
    metrics.update(intensity_mae=absolute / pixels, intensity_rmse=math.sqrt(squared / pixels),
                   samples=count, valid_pixels=pixels)
    if historical:
        metrics["intensity_weight"] = objective.active_intensity_weight
    return metrics, preview


def train(config_path, data_path, output, device="cpu", epochs=None,
          resume=None, init_checkpoint=None, threads=4):
    config_path = Path(config_path).resolve()
    data_path, output = Path(data_path).resolve(), Path(output).resolve()
    config = yaml.safe_load(config_path.read_text())
    recipe = config.get("recipe", "sparse-v1")
    if recipe not in {"sparse-v1", RECIPE}:
        raise ValueError("Unknown training recipe")
    historical = recipe == RECIPE
    model_path = (config_path.parent / config["model_config"]).resolve()
    model_config = yaml.safe_load(model_path.read_text())["model"]
    if model_config.get("intrinsic_net_type") != "unet" or model_config.get("intensity_head_type") != "deeplab":
        raise ValueError("This recipe requires the public U-Net + DeepLab architecture")
    settings = config["training"]
    seed = int(config.get("seed", 0))
    batch_size = int(settings["batch_size"])
    epoch_count = int(settings["epochs"] if epochs is None else epochs)
    if epoch_count < 1 or batch_size < 2:
        raise ValueError("epochs >= 1 and batch_size >= 2 are required")
    if resume and init_checkpoint:
        raise ValueError("Choose either resume or init_checkpoint")
    manifest = load_manifest(data_path)
    if manifest.get("recipe", "sparse-v1") != recipe:
        raise ValueError("Prepared target recipe differs from training recipe")
    train_set = IntensityDataset(data_path.parent, manifest, "train")
    val_set = IntensityDataset(data_path.parent, manifest, "val")
    if len(train_set) < 2:
        raise ValueError("Need at least two training samples for DeepLab BatchNorm")
    signature = {"model": model_config, "loss": config["loss"], "seed": seed,
                 "batch_size": batch_size, "lr": float(settings["lr"]),
                 "weight_decay": float(settings["weight_decay"]),
                 "data_sha256": sha256(data_path), "image_shape": manifest["image_shape"]}
    if historical:
        schedule_epochs = int(settings["epochs"])
        if epoch_count > schedule_epochs:
            raise ValueError("--epochs cannot extend the frozen historical cosine schedule")
        signature.update(recipe=recipe, schedule_epochs=schedule_epochs,
                         optimizer="AdamW", gradient_clip=1.0, drop_last=True)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = DeepIntrinsicIntensityNet(**model_config).to(device)
    optimizer_type = torch.optim.AdamW if historical else torch.optim.Adam
    optimizer = optimizer_type(model.parameters(), lr=signature["lr"],
                               weight_decay=signature["weight_decay"])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                  T_max=signature["schedule_epochs"], eta_min=signature["lr"] * 0.01)
                 if historical else None)
    start, best = 0, float("inf")
    history = output / "history.csv"
    if resume:
        checkpoint = load_checkpoint(resume)
        if checkpoint.get("signature") != signature:
            raise ValueError("Resume config/data mismatch; use init_checkpoint for a new run")
        start, best = int(checkpoint["epoch"]) + 1, float(checkpoint["best_val_total"])
        if start >= epoch_count:
            raise ValueError("--epochs is the total target epoch count and must exceed the checkpoint")
        if output.exists():
            if not history.is_file() or Path(resume).resolve() != output / "last.pth":
                raise ValueError("Resume in a new directory, or from this run's last.pth")
            with history.open() as stream:
                rows = list(csv.DictReader(stream))
            if not rows or int(rows[-1]["epoch"]) != start:
                raise ValueError("History/checkpoint mismatch; resume in a new output directory")
        else:
            # A new directory has no prior best checkpoint: select within this continuation.
            best = float("inf")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if historical:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            state = checkpoint["numpy_rng_state"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
            random.setstate(checkpoint["python_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    else:
        if output.exists():
            raise FileExistsError("Use a new output directory or explicitly --resume")
        if init_checkpoint:
            checkpoint = load_checkpoint(init_checkpoint)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "previews").mkdir(exist_ok=True)
    provenance = dict(signature, recipe=recipe, start_epoch=start,
                      target_epochs=epoch_count, torch_version=str(torch.__version__),
                      device=str(device), initialization_sha256=sha256(init_checkpoint) if init_checkpoint else None,
                      resume_sha256=sha256(resume) if resume else None,
                      best_selection_scope="completed epochs in this output directory",
                      note="Engineering training recipe, not exact reproduction of released weights")
    (output / f"run_from_epoch_{start:04d}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "model_config.yaml").write_text(yaml.safe_dump({"model": model_config}))
    for epoch in range(start, epoch_count):
        began = time.monotonic()
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loader = DataLoader(train_set, batch_sampler=epoch_batches(len(train_set), batch_size, seed, epoch,
                                                                         drop_last=historical),
                                  num_workers=0)
        train_metrics, _ = run_epoch(model, train_loader, device, config["loss"], optimizer, recipe, epoch)
        val_metrics, preview = run_epoch(model, DataLoader(val_set, batch_size=batch_size if historical else 1,
                                                          num_workers=0), device, config["loss"],
                                         recipe=recipe, epoch=epoch)
        if scheduler is not None:
            scheduler.step()
        improved = val_metrics["total"] < best
        best = min(best, val_metrics["total"])
        checkpoint = dict(epoch=epoch, model_state_dict=model.state_dict(),
                          optimizer_state_dict=optimizer.state_dict(), signature=signature,
                          best_val_total=best, torch_rng_state=torch.get_rng_state(),
                          cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                          train_metrics=train_metrics, val_metrics=val_metrics)
        if historical:
            state = np.random.get_state()
            checkpoint.update(scheduler_state_dict=scheduler.state_dict(),
                              numpy_rng_state=(state[0], state[1].tolist(), *state[2:]),
                              python_rng_state=random.getstate())
        save_checkpoint(output / "last.pth", checkpoint)
        inference_checkpoint = {"model_state_dict": model.state_dict(),
                                "model_config": model_config, "epoch": epoch}
        save_checkpoint(output / "last_inference.pth", inference_checkpoint)
        if improved:
            save_checkpoint(output / "best.pth", checkpoint)
            save_checkpoint(output / "best_inference.pth", inference_checkpoint)
        save_preview(output / "previews" / f"epoch_{epoch + 1:04d}.png", *preview, dense=historical)
        row = {"epoch": epoch + 1, "seconds": round(time.monotonic() - began, 3),
               **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        if historical:
            row.update(learning_rate=learning_rate, next_learning_rate=scheduler.get_last_lr()[0])
        exists = history.exists()
        with history.open("a", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, allow_nan=False), flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/config.yaml")
    parser.add_argument("--data", type=Path, required=True, help="Prepared manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, help="Total target epoch count, including before resume")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", type=Path)
    group.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    train(args.config, args.data, args.output, args.device, args.epochs,
          args.resume, args.init_checkpoint, args.threads)


if __name__ == "__main__":
    main()
