"""
Waymo 数据集封装
从 NCLR 项目提取并简化
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np


class WaymoDatasetWrapper:
    """
    Waymo 数据集的简化封装
    用于读取图像、点云、标定数据,并进行投影
    """

    def __init__(
        self,
        image_dir: Path,
        pointcloud_dir: Path,
        calib_dir: Optional[Path] = None,
        downsample_scale: float = 0.25,
        start_frame: int = 0,
        num_frames: Optional[int] = None,
        step: int = 1,
    ):
        """
        Args:
            image_dir: 图像目录 (包含 frame_XXXXXX_camY.jpg)
            pointcloud_dir: 点云目录 (包含 frame_XXXXXX_raw_pc.{ply|npz})
            calib_dir: 标定文件目录 (如果与image_dir不同)
            downsample_scale: 投影时的下采样比例 (for overlap mask)
            start_frame: 起始帧索引
            num_frames: 处理的帧数量 (None 表示全部)
            step: 采样步长，1表示连续采样，2表示每隔1个采样一次
        """
        self.image_dir = Path(image_dir)
        self.pointcloud_dir = Path(pointcloud_dir)
        self.calib_dir = Path(calib_dir) if calib_dir else self.image_dir
        self.downsample_scale = float(downsample_scale)

        # 发现可用帧
        all_frames = self._discover_frames()

        # 应用帧选择
        if start_frame > 0:
            all_frames = all_frames[start_frame:]
        if step > 1:
            all_frames = all_frames[::step]
        if num_frames is not None:
            all_frames = all_frames[:num_frames]

        self.frames = all_frames
        self.cameras_per_frame = self._discover_cameras()

        print(f"✓ 发现 {len(self.frames)} 帧数据")

    def _discover_frames(self) -> List[int]:
        """发现可用的帧ID"""
        frames = set()
        # 从点云JSON文件中发现帧
        for json_path in self.pointcloud_dir.glob("frame_*_lidar1.json"):
            stem = json_path.stem
            parts = stem.split("_")
            if len(parts) >= 2 and parts[0] == "frame":
                try:
                    frame_id = int(parts[1])
                    frames.add(frame_id)
                except ValueError:
                    continue
        return sorted(list(frames))

    def _discover_cameras(self) -> Dict[int, List[int]]:
        """发现每帧的相机ID"""
        cameras = {}
        for json_path in self.calib_dir.glob("frame_*_cam*.json"):
            stem = json_path.stem
            parts = stem.split("_")
            if len(parts) != 3 or parts[0] != "frame":
                continue
            try:
                frame_id = int(parts[1])
                cam_id = int(parts[2].replace("cam", ""))
                cameras.setdefault(frame_id, [])
                if cam_id not in cameras[frame_id]:
                    cameras[frame_id].append(cam_id)
            except ValueError:
                continue
        # 排序相机ID
        for fid in cameras:
            cameras[fid].sort()
        return cameras

    def get_frame_info(self, frame_id: int) -> Dict:
        """
        获取帧的基本信息

        Returns:
            info: {
                'frame_id': int,
                'cameras': [1, 2, 3, 4, 5],
                'image_paths': {cam_id: Path},
                'pointcloud_path': Path,
                'calib_paths': {cam_id: Path, 'lidar': Path}
            }
        """
        cam_ids = self.cameras_per_frame.get(frame_id, [])
        info = {
            'frame_id': frame_id,
            'cameras': cam_ids,
            'image_paths': {},
            'calib_paths': {},
        }

        # 图像路径
        for cam_id in cam_ids:
            img_path = self.image_dir / f"frame_{frame_id:06d}_cam{cam_id}.jpg"
            if img_path.exists():
                info['image_paths'][cam_id] = img_path

        # 点云路径 (优先npz,其次ply)
        pc_npz = self.pointcloud_dir / f"frame_{frame_id:06d}_raw_pc.npz"
        pc_ply = self.pointcloud_dir / f"frame_{frame_id:06d}_raw_pc.ply"
        if pc_npz.exists():
            info['pointcloud_path'] = pc_npz
        elif pc_ply.exists():
            info['pointcloud_path'] = pc_ply
        else:
            info['pointcloud_path'] = None

        # 标定路径
        for cam_id in cam_ids:
            calib_path = self.calib_dir / f"frame_{frame_id:06d}_cam{cam_id}.json"
            if calib_path.exists():
                info['calib_paths'][cam_id] = calib_path
        lidar_calib = self.pointcloud_dir / f"frame_{frame_id:06d}_lidar1.json"
        if lidar_calib.exists():
            info['calib_paths']['lidar'] = lidar_calib

        return info

    def load_image(self, frame_id: int, cam_id: int, color_mode='RGB') -> np.ndarray:
        """
        加载图像

        Args:
            frame_id: 帧ID
            cam_id: 相机ID
            color_mode: 'RGB' 或 'BGR'

        Returns:
            image: (H, W, 3), uint8
        """
        img_path = self.image_dir / f"frame_{frame_id:06d}_cam{cam_id}.jpg"
        if not img_path.exists():
            raise FileNotFoundError(f"图像不存在: {img_path}")

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"无法加载图像: {img_path}")

        if color_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_pointcloud(self, frame_id: int) -> np.ndarray:
        """
        加载点云

        Args:
            frame_id: 帧ID

        Returns:
            points: (N, 4), 列为 [x, y, z, intensity]
        """
        pc_npz = self.pointcloud_dir / f"frame_{frame_id:06d}_raw_pc.npz"
        pc_ply = self.pointcloud_dir / f"frame_{frame_id:06d}_raw_pc.ply"

        if pc_npz.exists():
            return self._load_npz(pc_npz)
        elif pc_ply.exists():
            return self._load_ply(pc_ply)
        else:
            raise FileNotFoundError(f"点云不存在: frame_{frame_id:06d}")

    def _load_npz(self, path: Path) -> np.ndarray:
        """从NPZ加载点云"""
        data = np.load(str(path))

        # 尝试多种格式
        if "points" in data and "intensity" in data:
            # 格式1: points (N,3) + intensity (N,) 分开存储
            xyz = data["points"]
            inten = data["intensity"].reshape(-1, 1)
            pts = np.concatenate([xyz, inten], axis=1)
        elif "points" in data:
            # 格式2: points (N,4) 已包含强度
            pts = data["points"]
        elif "xyz" in data:
            xyz = data["xyz"]
            if "intensity" in data:
                inten = data["intensity"].reshape(-1, 1)
                pts = np.concatenate([xyz, inten], axis=1)
            else:
                pts = xyz
        else:
            # 使用第一个数组
            key0 = list(data.keys())[0]
            pts = data[key0]

        # 确保形状为 (N, 4)
        if pts.shape[1] < 4:
            pad = np.zeros((pts.shape[0], 4 - pts.shape[1]), dtype=np.float32)
            pts = np.concatenate([pts, pad], axis=1)
        elif pts.shape[1] > 4:
            pts = pts[:, :4]

        return pts.astype(np.float32)

    def _load_ply(self, path: Path) -> np.ndarray:
        """从PLY加载点云 (简单的ASCII格式解析)"""
        with open(path, 'r') as f:
            line = f.readline().strip()
            if line != "ply":
                raise ValueError(f"不是PLY文件: {path}")

            # 跳过头部
            while line != "end_header":
                line = f.readline().strip()

            # 读取数据行
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rows.append([float(x) for x in parts])

        arr = np.asarray(rows, dtype=np.float32)

        # 确保形状为 (N, 4)
        if arr.shape[1] < 4:
            pad = np.zeros((arr.shape[0], 4 - arr.shape[1]), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > 4:
            arr = arr[:, :4]

        return arr

    def load_calibration(self, frame_id: int, cam_id: int) -> Dict:
        """
        加载相机标定数据

        Returns:
            calib: {
                'intrinsic': [fx, fy, cx, cy],
                'extrinsic': 4x4 numpy array (cam->ego),
                'width': int,
                'height': int
            }
        """
        calib_path = self.calib_dir / f"frame_{frame_id:06d}_cam{cam_id}.json"
        if not calib_path.exists():
            raise FileNotFoundError(f"标定文件不存在: {calib_path}")

        with open(calib_path, 'r') as f:
            data = json.load(f)

        calib = {
            'intrinsic': data['intrinsic'][:4],  # [fx, fy, cx, cy]
            'extrinsic': np.asarray(data['extrinsic'], dtype=np.float32).reshape(4, 4),
            'width': int(data['width']),
            'height': int(data['height']),
        }

        return calib

    def load_lidar_calibration(self, frame_id: int) -> Dict:
        """
        加载雷达标定数据

        Returns:
            calib: {
                'extrinsic': 4x4 numpy array (lidar->ego),
                'ego2world': 4x4 numpy array
            }
        """
        calib_path = self.pointcloud_dir / f"frame_{frame_id:06d}_lidar1.json"
        if not calib_path.exists():
            raise FileNotFoundError(f"雷达标定不存在: {calib_path}")

        with open(calib_path, 'r') as f:
            data = json.load(f)

        calib = {
            'extrinsic': np.asarray(data['extrinsic'], dtype=np.float32).reshape(4, 4),
            'ego2world': np.asarray(
                data.get('ego2world', np.eye(4).flatten().tolist()),
                dtype=np.float32
            ).reshape(4, 4),
        }

        return calib

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        for frame_id in self.frames:
            yield frame_id
