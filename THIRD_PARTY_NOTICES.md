# Third-Party Notices

Original Drive3R contributions are licensed under [CC BY-NC 4.0](./LICENSE).
Third-party portions retain their own copyright notices and license terms;
source attribution and accompanying license texts are listed below.

## marepo source attribution

`src/models/layers/camera_head.py` contains `ResConvBlock`, adapted via
[Pi3's camera head](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/pi3/models/layers/camera_head.py)
from [marepo's transformer](https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/transformer/transformer.py).
The original provenance comment and Niantic copyright notice are retained.

The full upstream terms are included in
[licenses/marepo-LICENSE.txt](./licenses/marepo-LICENSE.txt).

## Vendored source code

| Path in repo | Upstream | License | Notes |
| --- | --- | --- | --- |
| `src/models/dinov2/` | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | [Apache License 2.0](./src/models/dinov2/LICENSE) | The vendored DINOv2 backbone provides the ViT encoder. The upstream Apache 2.0 LICENSE is preserved at `src/models/dinov2/LICENSE`. |
| Pi3-derived model / decoder portions under `src/models/` | [yyfz/Pi3](https://github.com/yyfz/Pi3) | [BSD 3-Clause](./licenses/Pi3-BSD-3-Clause.txt) | The full copyright, conditions and disclaimer are included. Inherited third-party portions retain the additional terms listed here. |
| `ResConvBlock` in `src/models/layers/camera_head.py` | [nianticlabs/marepo](https://github.com/nianticlabs/marepo) via Pi3 | [marepo custom license](./licenses/marepo-LICENSE.txt) | Adapted via Pi3's camera head; the original source reference is retained in the file. |
| `src/models/layers/conv_head.py` | [microsoft/MoGe](https://github.com/microsoft/MoGe) | MIT, with [full upstream license file](./licenses/MoGe-LICENSE.txt) | Microsoft copyright is retained. The upstream file also includes terms for other MoGe components. |

Additional retained source notices:

- `src/models/layers/pos_embed.py` retains Naver copyright and CC BY-NC-SA
  4.0 notices; its file-level license remains applicable.
- `src/models/layers/prope.py` includes its MIT license and copyright notices.
- Apache-licensed attention / block components retain their headers; the
  included `src/models/dinov2/LICENSE` provides the Apache 2.0 text.

### License text snapshots

The following files contain unmodified upstream license texts at fixed
revisions.

| Local text | Upstream revision |
| --- | --- |
| [Pi3-BSD-3-Clause.txt](./licenses/Pi3-BSD-3-Clause.txt) | [Pi3 `9fa3ddb3f8d53041f8b2738df404f62223bbaa7b`](https://github.com/yyfz/Pi3/blob/9fa3ddb3f8d53041f8b2738df404f62223bbaa7b/LICENSE) |
| [MoGe-LICENSE.txt](./licenses/MoGe-LICENSE.txt) | [MoGe `74fbce054ebed49800de42d0ad0e83495065719a`](https://github.com/microsoft/MoGe/blob/74fbce054ebed49800de42d0ad0e83495065719a/LICENSE) |
| [marepo-LICENSE.txt](./licenses/marepo-LICENSE.txt) | [marepo `9a45e2bb07e5bb8cb997620088d352b439b13e0e`](https://github.com/nianticlabs/marepo/blob/9a45e2bb07e5bb8cb997620088d352b439b13e0e/LICENSE) |

## Optional runtime dependency

`scripts/generate_dense_gt.py` imports [microsoft/MoGe](https://github.com/microsoft/MoGe)
(MoGe-2) for depth inference and affine alignment. Installation is described
in the README; its upstream terms are included in
[licenses/MoGe-LICENSE.txt](./licenses/MoGe-LICENSE.txt).

## Pretrained checkpoint provenance

The released `drive3r.pt` checkpoint is distributed for non-commercial
research use. Its dataset provenance and terms are listed below.

| Dataset | License / Terms |
| --- | --- |
| [NuScenes](https://www.nuscenes.org/nuscenes#terms) | nuTonomy nuScenes Dataset License (non-commercial research) |
| [Argoverse 2 Sensor](https://www.argoverse.org/about.html#terms-of-use) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| [Waymo Open Dataset](https://waymo.com/open/terms/) | Waymo Open Dataset License Agreement (non-commercial research) |
| [Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) |

## Other prior work cited in the README

The Acknowledgements section of `README.md` further credits
[DUSt3R](https://github.com/naver/dust3r),
[MonST3R](https://github.com/Junyi42/monst3r),
[Spann3R](https://github.com/HengyiWang/spann3r),
[CUT3R](https://github.com/CUT3R/CUT3R),
and [StreamVGGT](https://github.com/wzzheng/StreamVGGT) for inspiring
this line of work.
