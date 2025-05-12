import argparse, time

from libraries.classes import GsPly, Cluster
from libraries.utilities import ExLog, setup_torch_and_random
from libraries.cliconfigs import VGBuildConfig


def VgBuild(vg_build_config: VGBuildConfig):
    # load 3DGS asset
    gsply: GsPly = GsPly(vg_build_config=vg_build_config)
    primitives: Cluster = gsply.read()

    # iteratively build all levels of detail
    ExLog(f"Start building LODs...")
    lods = primitives.buildAllLodLayers()
    ExLog(f"Finish building LODs.")

    # save V3DG bundle
    ExLog(f"Start saving LODs...")
    lods.saveBundle()
    ExLog(f"Finish saving LODs...")


if __name__ == "__main__":
    ExLog("START")
    print()

    setup_torch_and_random()

    parser = argparse.ArgumentParser()
    vg_build_config = VGBuildConfig(parser=parser)
    args = parser.parse_args()
    vg_build_config.extract(args=args)
    vg_build_config.process()

    time_start = time.perf_counter()
    VgBuild(vg_build_config=vg_build_config)
    time_end = time.perf_counter()
    duration = time_end - time_start
    with open(vg_build_config.OUTPUT_FOLDER_PATH / "_records.py", "w") as f:
        f.write(f"{duration=}\n")

    print()
    ExLog("END")
