import argparse, datetime, zoneinfo, pathlib

import numpy as np

from libraries.utilities import ExLog


class BaseConfig:
    # https://stackoverflow.com/a/52403318/14298786
    def StrToBool(value: str):
        if isinstance(value, bool):
            return value
        if value.lower() in {"false", "f", "0", "no", "n"}:
            return False
        elif value.lower() in {"true", "t", "1", "yes", "y"}:
            return True
        raise ValueError(f"{value} is not a valid boolean value")

    def __init__(self, parser: argparse.ArgumentParser):
        ExLog(f"default config: {vars(self)}")
        for key, default_value in vars(self).items():
            if type(default_value) == bool:
                parser.add_argument(
                    "--" + key,
                    type=BaseConfig.StrToBool,
                    default=default_value,
                )
            else:
                parser.add_argument(
                    "--" + key,
                    type=type(default_value),
                    default=default_value,
                )

    def extract(self, args: argparse.Namespace):
        for key, new_value in vars(args).items():
            if key in vars(self):
                old_value = getattr(self, key)
                if old_value != new_value:
                    ExLog(f"modified argument: {key} ({old_value} -> {new_value})")
                setattr(self, key, new_value)
        ExLog(f"modified config: {vars(self)}")


class VGBuildConfig(BaseConfig):
    def __init__(self, parser: argparse.ArgumentParser = None):
        # [asset]

        self.GS_ASSETS_FOLDER: str = "./_3dgs-assets"

        self.ASSET_KIND: str = "donut"
        self.ASSET_NAME: str = "donut"
        self.ASSET_GSPLY_FILENAME: str = "point_cloud/iteration_30000/fused.ply"

        # [method]

        # n_g_in_c
        self.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER: int = 4096
        # n_c_in_cg
        self.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP: int = 2
        self.BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER: int = 2

        self.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY: str = (
            "voxels+osss23+s216"
        )
        self.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION: bool = True
        self.SIMPLIFICATION_ITERATION: int = 640

        # loss
        self.SIMPLIFICATION_LOSS_LAMBDA_DSSIM: float = 0.2

        # learning rate
        self.SIMPLIFICATION_LEARNING_RATE_POSITION: float = 0.0000160
        self.SIMPLIFICATION_LEARNING_RATE_SCALE: float = 0.005
        self.SIMPLIFICATION_LEARNING_RATE_QUATERNION: float = 0.001
        self.SIMPLIFICATION_LEARNING_RATE_OPACITY: float = 0.05
        self.SIMPLIFICATION_LEARNING_RATE_SH0: float = 0.0025

        # [bundle]

        self.VG_BUNDLES_FOLDER: str = "./_v3dg-bundles"
        self.BUNDLE_CLUSTERS_NPZ_FILENAME: str = "clusters.npz"
        self.BUNDLE_GAUSSIANS_NPZ_FILENAME: str = "gaussians.npz"

        # [DEBUG]

        self.SAVE_IMAGES_DURING_OPTIMIZATION: bool = False

        # [ConfigBase]

        if parser != None:
            super().__init__(parser=parser)

    def process(self):
        # [time]

        self.TIME_PREFIX_STR = datetime.datetime.now(
            tz=zoneinfo.ZoneInfo("Asia/Shanghai")
        ).strftime("%y%m%d-%H%M%S")

        # [asset]

        self.ASSET_GSPLY_PATH: pathlib.Path = pathlib.Path(
            f"{self.GS_ASSETS_FOLDER}/{self.ASSET_KIND}/{self.ASSET_NAME}/{self.ASSET_GSPLY_FILENAME}"
        )

        # [bundle]

        self.BUNDLE_KIND: str = self.ASSET_KIND
        self.BUNDLE_NAME: str = (
            f"{self.ASSET_NAME}_{self.TIME_PREFIX_STR}_iter{self.SIMPLIFICATION_ITERATION}_nginc{self.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER}_ncincg{self.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP}{'' if self.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION else '_noscaleexpansion'}"
        )

        self.BUNDLE_FOLDER_PATH: pathlib.Path = pathlib.Path(
            f"{self.VG_BUNDLES_FOLDER}/{self.BUNDLE_KIND}/{self.BUNDLE_NAME}"
        )
        self.BUNDLE_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        self.BUNDLE_CLUSTERS_NPZ_PATH: pathlib.Path = (
            self.BUNDLE_FOLDER_PATH / self.BUNDLE_CLUSTERS_NPZ_FILENAME
        )
        self.BUNDLE_GAUSSIANS_NPZ_PATH: pathlib.Path = pathlib.Path(
            self.BUNDLE_FOLDER_PATH / self.BUNDLE_GAUSSIANS_NPZ_FILENAME
        )

        # [output]

        self.OUTPUT_FOLDER_PATH: pathlib.Path = pathlib.Path(
            f"./_output/_build/{self.TIME_PREFIX_STR}_build_{self.BUNDLE_KIND}_{self.BUNDLE_NAME}/"
        )
        self.OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

        # [config]

        ExLog(f"processed config: {vars(self)=}")


class VGSelectConfig(BaseConfig):
    def __init__(self, parser: argparse.ArgumentParser):
        # [asset]

        self.GS_ASSETS_FOLDER: str = "./_3dgs-assets"

        # [bundle]

        self.VG_BUNDLES_FOLDER: str = "./_v3dg-bundles"
        self.BUNDLE_CLUSTERS_NPZ_FILENAME: str = "clusters.npz"
        self.BUNDLE_GAUSSIANS_NPZ_FILENAME: str = "gaussians.npz"

        # [scene description]

        self.LAYOUT_DESCRIPTION_JSON = "./_layout-descriptions/DonutSea.json"

        # [method]

        self.TAU: float = 2048.0
        # self.TAUS: list[float] = []
        # self.TAUS: list[float] = [
        #     8192,
        #     4096,
        #     2048,
        #     1024,
        #     512,
        # ]
        self.TAUS: list[float] = np.linspace(start=0.0, stop=36500.0, num=120).tolist()

        # [output]

        self.SAVE_GS_LAYOUT_GAUSSIANS_PLY: bool = False
        self.SAVE_VG_LAYOUT_BUNDLE_FILES: bool = False

        self.DISABLE_VG_AND_ONLY_RENDER_GS: bool = False

        self.SAVE_ORIGINAL_4K_IMAGES: bool = False
        self.SAVE_IMAGES_CONTINUOUS: bool = False
        self.SAVE_IMAGES_COMPARISON: bool = True

        # [experiments]

        self.SAVE_METRICS_AT_DIFFERENT_DISTANCES: bool = True

        self.CALCULATE_PSNR_WITH_NVS_GT: bool = False

        self.GSPLAT_RADIUS_CLIP: bool = False

        self.SAVE_IMAGES_OF_CONTINUOUS_TAU: bool = False

        # [ConfigBase]

        super().__init__(parser=parser)

    def process(self):
        # [tau]

        if self.TAU != -1.0:
            self.TAUS = [self.TAU]

        # [time]

        self.TIME_PREFIX_STR = datetime.datetime.now(
            tz=zoneinfo.ZoneInfo("Asia/Shanghai")
        ).strftime("%y%m%d-%H%M%S")

        # [output]

        # TODO .split('.')[0] crops 0.0
        self.OUTPUT_FOLDER_PATH: pathlib.Path = pathlib.Path(
            f"./_output/{self.TIME_PREFIX_STR}_select_{self.LAYOUT_DESCRIPTION_JSON.split('/')[-1].split('.')[0]}_tau{self.TAU}"
        )
        self.OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        (self.OUTPUT_FOLDER_PATH / "images").mkdir(parents=True, exist_ok=True)
        if self.SAVE_GS_LAYOUT_GAUSSIANS_PLY:
            (self.OUTPUT_FOLDER_PATH / "plys").mkdir(parents=True, exist_ok=True)
        if self.SAVE_VG_LAYOUT_BUNDLE_FILES:
            (self.OUTPUT_FOLDER_PATH / "bundles").mkdir(parents=True, exist_ok=True)
            (self.OUTPUT_FOLDER_PATH / "bundles/vg-layout-bundle").mkdir(
                parents=True, exist_ok=True
            )

        # [config]

        ExLog(f"processed config: {vars(self)=}")
        print()
