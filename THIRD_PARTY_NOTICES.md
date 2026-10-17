# Third-Party Notices

The repository's [Apache-2.0 license](LICENSE) does not replace third-party
notices or grant rights to model weights, calibration artifacts or datasets.

## cs-stereo

`inference/src/models/stn_net.py` adapts the spectral translation network
from [cs-stereo](https://github.com/tiancheng-zhi/cs-stereo).
Copyright (c) 2019 Tiancheng Zhi. The original
[MIT license](LICENSES/CS_STEREO_MIT.txt) is included with the source package.
Preserve this notice and the MIT text when redistributing that component.

## Models And Data

Model and calibration distribution terms are documented separately in the
[artifact notices](https://huggingface.co/zhangjunjie587/NIDAR/blob/main/LICENSE_NOTICES.md).
No blanket commercial-use permission for weights or calibration is asserted.
The upstream STN code license does not establish the PittsStereo training-data
or trained-weight license.

Waymo and nuScenes input must be obtained from their official providers under
the applicable agreements; see [data access](docs/DATA_ACCESS.md). Dataset SDKs
are installed as separate dependencies, not redistributed inside this package.
Existing research visualizations retain their separate
[source and data terms](docs/assets/README.md).

Third-party dependencies retain their own licenses. Inclusion of this notice
is not a certification that every artifact is unrestricted or suitable for
commercial deployment.
