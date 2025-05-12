# Virtualized 3D Gaussians (V3DG)

[**Project Page**](https://xijie-yang.github.io/V3DG/)

## Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes

(SIGGRAPH 2025 Conference Paper)

[Xijie Yang](https://orcid.org/0009-0009-3076-2595), [Linning Xu](https://eveneveno.github.io/lnxu/), [Lihan Jiang](https://jianglh-whu.github.io/), [Dahua Lin](http://dahua.site/), [Bo Dai](https://daibo.info/)

![](./_media/framework.jpeg)

## Get Started

### Environment Setup

All experiments were conducted on a Linux server running Ubuntu 22.04 with CUDA 11.8. Other versions may also be compatible.

```bash
conda create -y -n V3DG python=3.12
conda activate V3DG

pip install torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install plyfile==1.1 einops==0.8.1 tqdm==4.67.1 scipy==1.15.2 gsplat==1.4.0 opencv-python==4.11.0.86
```

### Offline Build Stage for each single asset

Place 3DGS assets under [`./_3dgs-assets/`](./_3dgs-assets). The donut asset is already included. Additional assets used in the paper are available on [huggingface](https://huggingface.co/datasets/XijieYang/V3DG/tree/main).

To build an asset (e.g., the donut asset at [`./_3dgs-assets/donut/donut/point_cloud/iteration_30000/fused.ply`](./_3dgs-assets/donut/donut/point_cloud/iteration_30000/fused.ply):

```bash
# For quick testing: set simplification iteration to 16. Note: the first-time gsplat compilation takes a few minutes; build time ~10s.
python v3dg-build.py --ASSET_KIND donut --ASSET_NAME donut --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply --SIMPLIFICATION_ITERATION 16
# For default full build (640 iterations): build time ~100s.
python v3dg-build.py --ASSET_KIND donut --ASSET_NAME donut --ASSET_GSPLY_FILENAME point_cloud/iteration_30000/fused.ply
```

The build process generates a bundle of the asset in `./_v3dg-bundles/`, which includes clusters at multiple levels of detail.

To see all available parameters for [`v3dg-build.py`](./v3dg-build.py), refer to the `VGBuildConfig` class in [`./libraries/cliconfigs.py`](./libraries/cliconfigs.py).

To build all assets used in the paper, see the scripts in [`./_scripts/`](./_scripts/) ([`./_scripts/v3dg-build.sh`](./_scripts/v3dg-build.sh), [`./_scripts/v3dg-build-ablations.sh`](./_scripts/v3dg-build-ablations.sh), and [`./_scripts/v3dg-build-not-rectified-scenes.sh`](_scripts/v3dg-build-not-rectified-scenes.sh)).

Core code: [`./libraries/classes.py`](./libraries/classes.py)
`coarser_lod_layer = current_lod_layer.buildCoarserLodLayer()` in `Cluster.buildAllLodLayers()` -> `Clusters.buildCoarserLodLayer()`.

### Online Selection Stage for instances in the scene

To render composed scenes with 3DGS instances, first prepare a layout description file. Examples are available in [`./_layout-descriptions/`](./_layout-descriptions/), such as [`./_layout-descriptions/_SingleDonut.json`](./_layout-descriptions/_SingleDonut.json) and [`./_layout-descriptions/DonutSea.json`](./_layout-descriptions/DonutSea.json).

Since bundles are saved with timestamps (e.g. `donut_%y%m%d-%H%M%S_iter640_nginc4096_ncincg2`), make sure `bundle_name`s in the layout descriptions match accordingly.

To render scenes with vanilla 3DGS and our V3DG system:

```bash
# Test rendering a single donut at a fixed view.
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/_SingleDonut.json --TAU 2048 --SAVE_METRICS_AT_DIFFERENT_DISTANCES f
# Render the scene DonutSea (2048 donuts, 400 views). Full evaluation costs ~48 mins and reaches ~79 GB peak GPU memory (on NVIDIA A100, might not be the minimum need).
# Note: Scene composition is currently single-threaded and lacks a streaming module to optimize memory usage.
python v3dg-select.py --LAYOUT_DESCRIPTION_JSON ./_layout-descriptions/DonutSea.json --TAU 2048
```

Renders (approx. 5 GB per scene) are saved in `./_output/`, along with metrics recorded in `_metrics.py` and each filename.

To see all parameters for [`v3dg-select.py`](./v3dg-select.py), refer to `VGSelectConfig` in [`./libraries/cliconfigs.py`](./libraries/cliconfigs.py).

To render all composed scenes used in the paper, check [`./_scripts/`](./_scripts/) ([`./_scripts/v3dg-select.sh`](./_scripts/v3dg-select.sh), [`./_scripts/v3dg-select-ablations.sh`](./_scripts/v3dg-select-ablations.sh), and [`./_scripts/v3dg-select-comparisons.sh`](./_scripts/v3dg-select-comparisons.sh)). For rendering parts in the supplementart video, check [`./_layout-descriptions/_videos/`](./_layout-descriptions/_videos/).

Core code: [`./v3dg-select.py`](./v3dg-select.py) `vg_selected_gaussians = vg_layout_bundle.selectedGaussians(...)` in `VgSelect()` -> `VgBundle.selectGaussians()`.

## Branches

The `main` branch contains the exact code used for the reproducibility of all experiments and results presented in the paper. The `dev` branch hosts ongoing development and improvements of this project.

## BibTeX Citation

```bibtex
@inproceedings{Yang2025V3DG,
    author = {Yang, Xijie and Xu, Linning and Jiang, Lihan and Lin, Dahua and Dai, Bo},
    title = {Virtualized 3D Gaussians: Flexible Cluster-based Level-of-Detail System for Real-Time Rendering of Composed Scenes},
    booktitle = {ACM SIGGRAPH 2025 Conference Papers},
    year = {2025},
    doi = {10.1145/3721238.3730602},
    url = {https://doi.org/10.1145/3721238.3730602},
    publisher = {Association for Computing Machinery},
    series = {SIGGRAPH '25}
}
```

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/city-super/V3DG.svg?variant=adaptive)](https://starchart.cc/city-super/V3DG)
