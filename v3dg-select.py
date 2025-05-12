# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import pathlib, json, math, argparse, time

import numpy as np
import torch
import cv2
import einops

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
import scipy.spatial.transform


from libraries.utilities import (
    setup_torch_and_random,
    ExLog,
    ExTimer,
    UTILITY,
    UTILITY_PYTORCH,
    UTILITY_FLIP,
)
from libraries.classes import Camera, GsPly, Cluster, Clusters
from libraries.cliconfigs import VGBuildConfig, VGSelectConfig


def GsLoadLayoutGaussiansFromDisk(
    vg_select_config: VGSelectConfig,
    assets: list[dict],
) -> Cluster:
    loaded_instances = Clusters(clusters=[])

    for i_asset, current_asset in enumerate(assets):
        print()
        ExLog(
            f"Processing asset {i_asset}: {current_asset['asset_kind']}/{current_asset['asset_name']}/{current_asset['ply_suffix']}..."
        )

        vg_build_config = VGBuildConfig()
        vg_build_config.ASSET_GSPLY_PATH = pathlib.Path(
            f"{vg_select_config.GS_ASSETS_FOLDER}/{current_asset['asset_kind']}/{current_asset['asset_name']}/{current_asset['ply_suffix']}"
        )
        current_asset_gaussians: Cluster = GsPly(vg_build_config=vg_build_config).read()

        # [compose the scene from instances of current asset]

        current_asset_instances: list[dict] = current_asset["instances"]
        for i_instance, current_instance in enumerate(current_asset_instances):
            # print()
            # ExLog(
            #     f"Start processing instance {i_instance}: {current_instance['instance_name']}..."
            # )
            current_instance_gaussians = current_asset_gaussians.transformed(
                scale_factor=current_instance["scale"],
                R=torch.from_numpy(
                    scipy.spatial.transform.Rotation.from_quat(
                        current_instance["quaternions"],
                        scalar_first=True,
                    ).as_matrix()
                ),
                t=torch.tensor(current_instance["positions"]),
            )
            loaded_instances.append(current_instance_gaussians)

    return loaded_instances.consolidateIntoASingleCluster()


class VgBundle:
    def Load(
        vg_select_config: VGSelectConfig,
        bundle_kind: str,
        bundle_name: str,
    ) -> VgBundle:
        # [read gaussians.npz of current_asset]

        gaussians_npz_path: pathlib.Path = pathlib.Path(
            f"{vg_select_config.VG_BUNDLES_FOLDER}/{bundle_kind}/{bundle_name}/{vg_select_config.BUNDLE_GAUSSIANS_NPZ_FILENAME}"
        )
        gaussians_npz_data: np.ndarray = np.load(gaussians_npz_path)
        gaussians_count: int = gaussians_npz_data["positions"].shape[0]
        ExLog(f"Read {gaussians_count} points from {gaussians_npz_path}.")

        gaussians_positions = torch.from_numpy(gaussians_npz_data["positions"]).to(
            device="cuda",
            dtype=torch.float32,
        )
        gaussians_scales = torch.from_numpy(gaussians_npz_data["scales"]).to(
            device="cuda",
            dtype=torch.float32,
        )
        gaussians_quaternions = torch.from_numpy(gaussians_npz_data["quaternions"]).to(
            device="cuda",
            dtype=torch.float32,
        )
        gaussians_opacities = torch.from_numpy(gaussians_npz_data["opacities"]).to(
            device="cuda",
            dtype=torch.float32,
        )
        gaussians_rgbs = torch.from_numpy(gaussians_npz_data["rgbs"]).to(
            device="cuda",
            dtype=torch.float32,
        )

        # [read clusters.npz of current_asset]

        clusters_npz_path: pathlib.Path = pathlib.Path(
            f"{vg_select_config.VG_BUNDLES_FOLDER}/{bundle_kind}/{bundle_name}/{vg_select_config.BUNDLE_CLUSTERS_NPZ_FILENAME}"
        )
        clusters_npz_data: np.ndarray = np.load(clusters_npz_path)
        clusters_count: int = clusters_npz_data["lod_levels"].shape[0]
        ExLog(f"Read {clusters_count} clusters from {clusters_npz_path}.")

        clusters_lod_levels = torch.from_numpy(clusters_npz_data["lod_levels"]).to(
            device="cuda",
            dtype=torch.int32,
        )
        clusters_start_indices = torch.from_numpy(
            clusters_npz_data["start_indices"]
        ).to(
            device="cuda",
            dtype=torch.int32,
        )
        clusters_gaussians_counts = torch.from_numpy(clusters_npz_data["counts"]).to(
            device="cuda",
            dtype=torch.int32,
        )
        clusters_child_centers = torch.from_numpy(
            clusters_npz_data["child_centers"]
        ).to(
            device="cuda",
            dtype=torch.float32,
        )
        clusters_child_radii = torch.from_numpy(clusters_npz_data["child_radii"]).to(
            device="cuda",
            dtype=torch.float32,
        )
        clusters_parent_centers = torch.from_numpy(
            clusters_npz_data["parent_centers"]
        ).to(
            device="cuda",
            dtype=torch.float32,
        )
        clusters_parent_radii = torch.from_numpy(clusters_npz_data["parent_radii"]).to(
            device="cuda",
            dtype=torch.float32,
        )

        return VgBundle(
            gaussians_count=gaussians_count,
            gaussians_positions=gaussians_positions,
            gaussians_scales=gaussians_scales,
            gaussians_quaternions=gaussians_quaternions,
            gaussians_opacities=gaussians_opacities,
            gaussians_rgbs=gaussians_rgbs,
            clusters_count=clusters_count,
            clusters_lod_levels=clusters_lod_levels,
            clusters_start_indices=clusters_start_indices,
            clusters_gaussians_counts=clusters_gaussians_counts,
            clusters_child_centers=clusters_child_centers,
            clusters_child_radii=clusters_child_radii,
            clusters_parent_centers=clusters_parent_centers,
            clusters_parent_radii=clusters_parent_radii,
        )

    def TransformGaussians(
        gaussians_count: int,
        gaussians_positions: torch.Tensor,
        gaussians_scales: torch.Tensor,
        gaussians_quaternions: torch.Tensor,
        scale_factor: float,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        Rt_matrices_original = torch.zeros((gaussians_count, 4, 4), dtype=torch.float32)
        Rt_matrices_original[:, 3, 3] = 1.0
        Rt_matrices_original[:, :3, 3] = gaussians_positions
        # NOTICE: r xyz -> xyz w
        Rt_matrices_original[:, :3, :3] = torch.from_numpy(
            scipy.spatial.transform.Rotation.from_quat(
                gaussians_quaternions.cpu().numpy(),
                scalar_first=True,
            ).as_matrix()
        )

        # random scale (notice that we only use uniform scaling on xyz the same time)
        S1 = torch.eye(4, dtype=torch.float32)
        S1[:3] *= scale_factor
        new_scales = gaussians_scales * scale_factor

        # random rotation
        R1 = torch.eye(4, dtype=torch.float32)
        R1[:3, :3] = R

        # layout tables
        T1 = torch.eye(4, dtype=torch.float32)
        T1[:3, 3] = t

        # column vector, from right to left
        # the order of R1 and S1 can be changed, but T1 must be after R1
        Rt_full = S1 @ R1

        Rt_matrices_new = T1 @ Rt_full @ Rt_matrices_original
        new_positions = Rt_matrices_new[:, :3, 3]
        # NOTICE: xyz w -> r xyz
        new_quaternions = torch.from_numpy(
            scipy.spatial.transform.Rotation.from_matrix(
                (R1 @ Rt_matrices_original)[:, :3, :3].cpu().numpy()
            ).as_quat(scalar_first=True),
        ).to(device="cuda", dtype=torch.float32)

        return (
            new_positions,
            new_scales,
            new_quaternions,
        )

    def TransformClustersCenters(
        clusters_count: int,
        # (3,)
        clusters_centers: torch.Tensor,
        scale_factor: float,
        # (3,3)
        R: torch.Tensor,
        # (3,)
        t: torch.Tensor,
    ) -> torch.Tensor:
        Rt_matrices_original = torch.zeros((clusters_count, 4, 4), dtype=torch.float32)
        Rt_matrices_original[:, 3, 3] = 1.0
        Rt_matrices_original[:, 3, :3] = clusters_centers
        Rt_matrices_original[:, :3, :3] = torch.eye(3)

        # random scale (notice that we only use uniform scaling on xyz the same time)
        S1 = torch.eye(4, dtype=torch.float32)
        S1[:3] *= scale_factor

        # random rotation
        R1 = torch.eye(4, dtype=torch.float32)
        R1[:3, :3] = R

        # layout tables
        T1 = torch.eye(4, dtype=torch.float32)
        T1[3, :3] = t

        # from left to right
        Rt_full = S1 @ R1 @ T1

        Rt_matrices_new = Rt_matrices_original @ Rt_full
        new_positions = Rt_matrices_new[:, 3, :3]

        return new_positions

    def __init__(
        self,
        gaussians_count: int,
        gaussians_positions: torch.Tensor,
        gaussians_scales: torch.Tensor,
        gaussians_quaternions: torch.Tensor,
        gaussians_opacities: torch.Tensor,
        gaussians_rgbs: torch.Tensor,
        clusters_count: int,
        clusters_lod_levels: torch.Tensor,
        clusters_start_indices: torch.Tensor,
        clusters_gaussians_counts: torch.Tensor,
        clusters_child_centers: torch.Tensor,
        clusters_child_radii: torch.Tensor,
        clusters_parent_centers: torch.Tensor,
        clusters_parent_radii: torch.Tensor,
    ) -> None:
        self.gaussians_count = gaussians_count
        self.gaussians_positions = gaussians_positions
        self.gaussians_scales = gaussians_scales
        self.gaussians_quaternions = gaussians_quaternions
        self.gaussians_opacities = gaussians_opacities
        self.gaussians_rgbs = gaussians_rgbs

        self.clusters_count = clusters_count
        self.clusters_lod_levels = clusters_lod_levels
        self.clusters_start_indices = clusters_start_indices
        self.clusters_gaussians_counts = clusters_gaussians_counts
        self.clusters_child_centers = clusters_child_centers
        self.clusters_child_radii = clusters_child_radii
        self.clusters_parent_centers = clusters_parent_centers
        self.clusters_parent_radii = clusters_parent_radii

    def transformed(
        self,
        scale_factor: float,
        # (3,3)
        R: torch.Tensor,
        # (3,)
        t: torch.Tensor,
    ) -> VgBundle:
        # [transform gaussians]

        (
            transformed_gaussians_positions,
            transformed_gaussians_scales,
            transformed_gaussians_quaternions,
        ) = VgBundle.TransformGaussians(
            gaussians_count=self.gaussians_count,
            gaussians_positions=self.gaussians_positions,
            gaussians_scales=self.gaussians_scales,
            gaussians_quaternions=self.gaussians_quaternions,
            scale_factor=scale_factor,
            R=R,
            t=t,
        )

        # [transform clusters]

        transformed_clusters_child_centers = VgBundle.TransformClustersCenters(
            clusters_count=self.clusters_count,
            clusters_centers=self.clusters_child_centers,
            scale_factor=scale_factor,
            R=R,
            t=t,
        )
        transformed_clusters_child_radii = self.clusters_child_radii * scale_factor
        transformed_clusters_parent_centers = VgBundle.TransformClustersCenters(
            clusters_count=self.clusters_count,
            clusters_centers=self.clusters_parent_centers,
            scale_factor=scale_factor,
            R=R,
            t=t,
        )
        transformed_clusters_parent_radii = self.clusters_parent_radii * scale_factor

        # [return]

        return VgBundle(
            gaussians_count=self.gaussians_count,
            gaussians_positions=transformed_gaussians_positions,
            gaussians_scales=transformed_gaussians_scales,
            gaussians_quaternions=transformed_gaussians_quaternions,
            # there might be no reason to clone since I only need to read the values?
            gaussians_opacities=self.gaussians_opacities.clone(),
            gaussians_rgbs=self.gaussians_rgbs,
            clusters_count=self.clusters_count,
            clusters_lod_levels=self.clusters_lod_levels.clone(),
            clusters_start_indices=self.clusters_start_indices.clone(),
            clusters_gaussians_counts=self.clusters_gaussians_counts.clone(),
            clusters_child_centers=transformed_clusters_child_centers,
            clusters_child_radii=transformed_clusters_child_radii,
            clusters_parent_centers=transformed_clusters_parent_centers,
            clusters_parent_radii=transformed_clusters_parent_radii,
        )

    def selectedGaussians(self, camera: Camera, tau: float) -> Cluster:
        # [calculate footprints of all clusters]

        # homogenous coordinates
        child_centers_3d = torch.ones((self.clusters_count, 4), dtype=torch.float32)
        child_centers_3d[:, :3] = self.clusters_child_centers
        child_centers_viewed = child_centers_3d @ camera.view_matrix
        cluster_child_footprint = (
            math.pi
            * self.clusters_child_radii**2
            / (child_centers_viewed[:, [2]] ** 2)
            * camera.focal_x
            * camera.focal_y
        )
        cluster_child_footprint[self.clusters_child_radii < 0] = -1

        parent_centers_3d = torch.ones((self.clusters_count, 4), dtype=torch.float32)
        parent_centers_3d[:, :3] = self.clusters_parent_centers
        parent_centers_viewed = parent_centers_3d @ camera.view_matrix
        cluster_parent_footprint = (
            math.pi
            * self.clusters_parent_radii**2
            / (parent_centers_viewed[:, [2]] ** 2)
            * camera.focal_x
            * camera.focal_y
        )
        cluster_parent_footprint[self.clusters_parent_radii == math.inf] = math.inf

        # [select clusters by tolerance]

        # (N_clusters, 1)
        selected_mask_cluster_tau = (cluster_child_footprint <= tau) & (
            cluster_parent_footprint > tau
        )

        # (N_clusters, 1)
        # selected_clusters_indices = (
        #     selected_mask_cluster_center_in_frustum & selected_mask_cluster_tau
        # )
        selected_clusters_indices = selected_mask_cluster_tau

        # ExLog(
        #     f"all_clusters-{self.clusters_count}-100.00%    clusters_tau-{selected_mask_cluster_tau.sum().item()}-{selected_mask_cluster_tau.sum().item()/self.clusters_count*100:.2f}%    clusters_both-{selected_clusters_indices.sum().item()}-{selected_clusters_indices.sum().item()/self.clusters_count*100:.2f}%",
        #     "DEBUG",
        # )

        # selected_clusters_count = selected_clusters_indices.sum().item()
        # ExLog(
        #     f"Select {selected_clusters_count} clusters ({(selected_clusters_count/self.clusters_count)*100:.2f}%) in the given camera view.",
        #     "DEBUG",
        # )

        selected_clusters_start_indices = self.clusters_start_indices[
            selected_clusters_indices[:, 0]
        ]
        selected_clusters_gaussians_counts = self.clusters_gaussians_counts[
            selected_clusters_indices[:, 0]
        ]
        selected_clusters_gaussians_indices: torch.Tensor = (
            UTILITY_PYTORCH.generate_indices(
                starts=selected_clusters_start_indices[:, 0],
                counts=selected_clusters_gaussians_counts[:, 0],
            )
        )

        return Cluster(
            count=selected_clusters_gaussians_indices.shape[0],
            positions=self.gaussians_positions[selected_clusters_gaussians_indices, :],
            scales=self.gaussians_scales[selected_clusters_gaussians_indices, :],
            quaternions=self.gaussians_quaternions[
                selected_clusters_gaussians_indices, :
            ],
            opacities=self.gaussians_opacities[selected_clusters_gaussians_indices, :],
            rgbs=self.gaussians_rgbs[selected_clusters_gaussians_indices, :],
        )

    def save(self, bundle_folder_path: pathlib.Path) -> None:
        # [save clusters.npz]

        np.savez(
            bundle_folder_path / "clusters.npz",
            lod_levels=self.clusters_lod_levels.cpu().numpy(),
            start_indices=self.clusters_start_indices.cpu().numpy(),
            counts=self.clusters_gaussians_counts.cpu().numpy(),
            child_centers=self.clusters_child_centers.cpu().numpy(),
            parent_centers=self.clusters_parent_centers.cpu().numpy(),
            child_radii=self.clusters_child_radii.cpu().numpy(),
            parent_radii=self.clusters_parent_radii.cpu().numpy(),
        )

        # [save gaussians.npz]

        np.savez(
            bundle_folder_path / "gaussians.npz",
            positions=self.gaussians_positions.cpu().numpy(),
            scales=self.gaussians_positions.cpu().numpy(),
            quaternions=self.gaussians_quaternions.cpu().numpy(),
            opacities=self.gaussians_opacities.cpu().numpy(),
            rgbs=self.gaussians_rgbs.cpu().numpy(),
        )


class VgBundles:
    def __init__(
        self,
        bundles: list[VgBundle],
    ) -> None:
        self.bundles = bundles
        self.count = len(self.bundles)

    def append(self, bundle: VgBundle) -> None:
        self.bundles.append(bundle)
        self.count = len(self.bundles)

    def consolidateIntoASingleVgBundle(self) -> VgBundle:
        bundles_gaussians_counts_right_shift = torch.tensor(
            [0.0] + [b.gaussians_count for b in self.bundles][:-1],
            dtype=torch.int32,
        )
        bundles_gaussians_offset = torch.cumsum(
            bundles_gaussians_counts_right_shift, dim=0
        )
        clusters_start_indices: torch.Tensor = torch.cat(
            [
                b.clusters_start_indices + bundles_gaussians_offset[i_b]
                for i_b, b in enumerate(self.bundles)
            ],
            dim=0,
        )

        return VgBundle(
            gaussians_count=sum([b.gaussians_count for b in self.bundles]),
            gaussians_positions=torch.cat(
                [b.gaussians_positions for b in self.bundles], dim=0
            ),
            gaussians_scales=torch.cat(
                [b.gaussians_scales for b in self.bundles], dim=0
            ),
            gaussians_quaternions=torch.cat(
                [b.gaussians_quaternions for b in self.bundles], dim=0
            ),
            gaussians_opacities=torch.cat(
                [b.gaussians_opacities for b in self.bundles], dim=0
            ),
            gaussians_rgbs=torch.cat([b.gaussians_rgbs for b in self.bundles], dim=0),
            clusters_count=sum([b.clusters_count for b in self.bundles]),
            clusters_lod_levels=torch.cat(
                [b.clusters_lod_levels for b in self.bundles], dim=0
            ),
            clusters_start_indices=clusters_start_indices,
            clusters_gaussians_counts=torch.cat(
                [b.clusters_gaussians_counts for b in self.bundles], dim=0
            ),
            clusters_child_centers=torch.cat(
                [b.clusters_child_centers for b in self.bundles], dim=0
            ),
            clusters_child_radii=torch.cat(
                [b.clusters_child_radii for b in self.bundles], dim=0
            ),
            clusters_parent_centers=torch.cat(
                [b.clusters_parent_centers for b in self.bundles], dim=0
            ),
            clusters_parent_radii=torch.cat(
                [b.clusters_parent_radii for b in self.bundles], dim=0
            ),
        )


def VgLoadLayoutBundleFromDisk(
    vg_select_config: VGSelectConfig,
    assets: list[dict],
) -> VgBundle:
    all_instances_bundles: VgBundles = VgBundles(bundles=[])

    for i_asset, current_asset in enumerate(assets):
        print()
        ExLog(
            f"Processing asset {i_asset}: {current_asset['bundle_kind']}/{current_asset['bundle_name']}..."
        )

        # [load bundle of current_asset]

        current_asset_bundle: VgBundle = VgBundle.Load(
            vg_select_config=vg_select_config,
            bundle_kind=current_asset["bundle_kind"],
            bundle_name=current_asset["bundle_name"],
        )

        # [create all clusters of instances of current_asset]

        current_asset_instances: list[dict] = current_asset["instances"]

        for i_instance, current_instance in enumerate(current_asset_instances):
            # print()
            # ExLog(
            #     f"Start processing instance {i_instance}: {current_instance['instance_name']}..."
            # )

            current_instance_scale_factor = current_instance["scale"]
            current_instance_R = torch.from_numpy(
                scipy.spatial.transform.Rotation.from_quat(
                    current_instance["quaternions"],
                    scalar_first=True,
                ).as_matrix()
            ).to(device="cuda", dtype=torch.float32)
            current_instance_t = torch.from_numpy(
                np.array(current_instance["positions"])
            ).to(device="cuda", dtype=torch.float32)

            transformed_bundle = current_asset_bundle.transformed(
                scale_factor=current_instance_scale_factor,
                R=current_instance_R,
                t=current_instance_t,
            )

            all_instances_bundles.append(bundle=transformed_bundle)
    return all_instances_bundles.consolidateIntoASingleVgBundle()


def VgSelect(vg_select_config: VGSelectConfig):
    ExLog(f"Read layout description...")
    with open(vg_select_config.LAYOUT_DESCRIPTION_JSON) as f:
        layout_description = json.load(f)
    layouts: list[dict] = layout_description["layouts"]

    i_frame_offset: int = 0
    for i_layout, current_layout in enumerate(layouts):
        print()
        ExLog(f"Processing layout {i_layout}: {current_layout['layout_name']}...")

        current_cameras: list[dict] = current_layout["cameras"]

        current_assets: list[dict] = current_layout["assets"]

        # [gs: read instances of assets from disk for current layout]
        # [gs: consolidate gaussians for current layout]

        with ExTimer("GsLoadLayoutGaussiansFromDisk()"):
            gs_layout_gaussians = GsLoadLayoutGaussiansFromDisk(
                vg_select_config=vg_select_config,
                assets=current_assets,
            )
        ExLog(f"{gs_layout_gaussians.count=}")
        if vg_select_config.SAVE_GS_LAYOUT_GAUSSIANS_PLY:
            gs_layout_gaussians.saveOriginalPly(
                path=vg_select_config.OUTPUT_FOLDER_PATH
                / "plys/gs_layout_gaussians.ply"
            )

        # [vg: read clusters of all instances of all assets from disk for current layout]

        if not vg_select_config.DISABLE_VG_AND_ONLY_RENDER_GS:
            with ExTimer("VgLoadClustersFromDisk()"):
                vg_layout_bundle = VgLoadLayoutBundleFromDisk(
                    vg_select_config=vg_select_config,
                    assets=current_assets,
                )
            ExLog(f"{vg_layout_bundle.gaussians_count=}")
            if vg_select_config.SAVE_VG_LAYOUT_BUNDLE_FILES:
                vg_layout_bundle.save(
                    bundle_folder_path=vg_select_config.OUTPUT_FOLDER_PATH
                    / "bundles/vg-layout-bundle/"
                )

        # [metrics: only for the first layout, record metrics]

        metrics_gs_number_of_gaussians_in_frustum: list[float] = []
        metrics_vg_number_of_gaussians_in_frustum: list[float] = []

        metrics_gs_duration_all: list[float] = []
        metrics_vg_duration_select: list[float] = []
        metrics_vg_duration_render: list[float] = []
        metrics_vg_duration_all: list[float] = []

        metrics_gs_fps: list[float] = []
        metrics_vg_fps: list[float] = []

        metrics_gs_over_gs_SSAA4K_flip: list[float] = []
        metrics_vg_over_gs_SSAA4K_flip: list[float] = []

        if vg_select_config.CALCULATE_PSNR_WITH_NVS_GT:
            metrics_gs_psnr_nvs_gt: list[float] = []
            metrics_vg_psnr_nvs_gt: list[float] = []

        # i_camera is a local counter in current_layout
        for i_camera, current_camera in enumerate(current_cameras):
            # i_frame is a global counter in the layout description
            i_frame = i_frame_offset + i_camera

            print()
            ExLog(
                f"Processing layout {i_layout} camera {i_camera} frame {i_frame}: {current_camera['camera_name']}..."
            )

            # [read camera]

            camera_width: int = current_camera["camera_width"]
            camera_height: int = current_camera["camera_height"]
            camera_angle_x: float = current_camera["camera_angle_x"]
            camera_angle_y: float = current_camera["camera_angle_y"]
            camera_tranform_matrix: list[list[float]] = current_camera[
                "transform_matrix"
            ]

            if vg_select_config.CALCULATE_PSNR_WITH_NVS_GT:
                camera_gt_path = current_camera["gt_path"]
                camera_gt_image = UTILITY.ReadImage(camera_gt_path)

            camera_R = torch.zeros((3, 3), device="cuda", dtype=torch.float32)
            camera_R[0] = torch.tensor(camera_tranform_matrix[0])[:3]
            camera_R[1] = torch.tensor(camera_tranform_matrix[1])[:3]
            camera_R[2] = torch.tensor(camera_tranform_matrix[2])[:3]
            camera_R[:, 1:3] *= -1.0  # convert Blender(OpenGL) R to COLMAP R

            camera_t = torch.zeros((3,), device="cuda", dtype=torch.float32)
            camera_t[0] = torch.tensor(camera_tranform_matrix[0])[3]
            camera_t[1] = torch.tensor(camera_tranform_matrix[1])[3]
            camera_t[2] = torch.tensor(camera_tranform_matrix[2])[3]

            current_camera_instance_1080p = Camera(
                image_width=camera_width,
                image_height=camera_height,
                focal_x=camera_width / 2 / math.tan(camera_angle_x / 2),
                focal_y=camera_height / 2 / math.tan(camera_angle_y / 2),
                R=camera_R,
                t=camera_t,
            )
            current_camera_instance_4K = Camera(
                image_width=camera_width * 2,
                image_height=camera_height * 2,
                focal_x=camera_width * 2 / 2 / math.tan(camera_angle_x / 2),
                focal_y=camera_height * 2 / 2 / math.tan(camera_angle_y / 2),
                R=camera_R,
                t=camera_t,
            )

            # [gs: for a given camera, render composed gaussians]

            # 1080p

            (
                rendered_image_gs_1080p,
                count_infrustum_gs_1080p,
                _,
                time_duration_gs_render_1080p,
            ) = gs_layout_gaussians.renderReturnCountAndDuration(
                camera=current_camera_instance_1080p
            )

            if vg_select_config.CALCULATE_PSNR_WITH_NVS_GT:
                psnr_value_gs_gt = UTILITY.Psnr(
                    rendered_image_gs_1080p[:3] * rendered_image_gs_1080p[3],
                    (
                        camera_gt_image[:3] * camera_gt_image[3]
                        if camera_gt_image.shape[0] == 4
                        else camera_gt_image
                    ),
                ).item()
                metrics_gs_psnr_nvs_gt.append(psnr_value_gs_gt)

            # 4K

            (
                rendered_image_gs_4K,
                count_infrustum_gs_4K,
                _,
                time_duration_gs_render_4K,
            ) = gs_layout_gaussians.renderReturnCountAndDuration(
                camera=current_camera_instance_4K
            )

            if vg_select_config.SAVE_ORIGINAL_4K_IMAGES:
                if vg_select_config.SAVE_IMAGES_COMPARISON:
                    UTILITY.SaveImage(
                        rendered_image_gs_4K,
                        path=vg_select_config.OUTPUT_FOLDER_PATH
                        / f"images/{i_frame}_gs-4K_count{count_infrustum_gs_4K}-{count_infrustum_gs_4K/gs_layout_gaussians.count*100:.2f}%-count{gs_layout_gaussians.count}_fps{1.0 / time_duration_gs_render_4K:.2f}.png",
                    )
                if vg_select_config.SAVE_IMAGES_CONTINUOUS:
                    UTILITY.SaveImage(
                        rendered_image_gs_4K,
                        path=vg_select_config.OUTPUT_FOLDER_PATH
                        / f"images/gs-4K_{i_frame}_count{count_infrustum_gs_4K}-{count_infrustum_gs_4K/gs_layout_gaussians.count*100:.2f}%-count{gs_layout_gaussians.count}_fps{1.0 / time_duration_gs_render_4K:.2f}.png",
                    )

            # SSAA

            rendered_image_gs_SSAA4K = einops.rearrange(
                torch.tensor(
                    cv2.resize(
                        einops.rearrange(rendered_image_gs_4K, "c h w -> h w c")
                        .cpu()
                        .numpy(),
                        (
                            int(rendered_image_gs_4K.shape[2] / 2),
                            int(rendered_image_gs_4K.shape[1] / 2),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                ),
                "h w c -> c h w",
            )

            if vg_select_config.SAVE_IMAGES_COMPARISON:
                UTILITY.SaveImage(
                    rendered_image_gs_SSAA4K,
                    path=vg_select_config.OUTPUT_FOLDER_PATH
                    / f"images/{i_frame}_gs-SSAA4K.png",
                )
            if vg_select_config.SAVE_IMAGES_CONTINUOUS:
                UTILITY.SaveImage(
                    rendered_image_gs_SSAA4K,
                    path=vg_select_config.OUTPUT_FOLDER_PATH
                    / f"images/gs-SSAA4K_{i_frame}.png",
                )

            # flip

            flip_values_gs_1080p: torch.Tensor = UTILITY_FLIP.CalculateFlipValues(
                image_test=rendered_image_gs_1080p[:3] * rendered_image_gs_1080p[3],
                image_reference=rendered_image_gs_SSAA4K[:3]
                * rendered_image_gs_SSAA4K[3],
            )
            flip_value_gs_1080p = flip_values_gs_1080p.mean().item()

            image_flip_gs_1080p = UTILITY_FLIP.FlipValuesToRgbHeatMap(
                flip_values_gs_1080p
            )

            if vg_select_config.SAVE_IMAGES_COMPARISON:
                UTILITY.SaveImage(
                    rendered_image_gs_1080p,
                    path=vg_select_config.OUTPUT_FOLDER_PATH
                    / f"images/{i_frame}_gs_count{count_infrustum_gs_1080p}-{count_infrustum_gs_1080p/gs_layout_gaussians.count*100:.2f}%-count{gs_layout_gaussians.count}_flip{flip_value_gs_1080p:.4f}_fps{1.0 / time_duration_gs_render_1080p:.2f}.png",
                )
                UTILITY.SaveImage(
                    image_flip_gs_1080p,
                    path=vg_select_config.OUTPUT_FOLDER_PATH
                    / f"images/{i_frame}_gs-flip_{flip_value_gs_1080p:.4f}.png",
                )
            if vg_select_config.SAVE_IMAGES_CONTINUOUS:
                UTILITY.SaveImage(
                    rendered_image_gs_1080p,
                    path=vg_select_config.OUTPUT_FOLDER_PATH
                    / f"images/gs_{i_frame}_count{count_infrustum_gs_1080p}-{count_infrustum_gs_1080p/gs_layout_gaussians.count*100:.2f}%-count{gs_layout_gaussians.count}_flip{flip_value_gs_1080p:.4f}_fps{1.0 / time_duration_gs_render_1080p:.2f}.png",
                )
                UTILITY.SaveImage(
                    image_flip_gs_1080p,
                    path=vg_select_config.OUTPUT_FOLDER_PATH
                    / f"images/gs-flip_{i_frame}_{flip_value_gs_1080p:.4f}.png",
                )

            if vg_select_config.SAVE_IMAGES_OF_CONTINUOUS_TAU:
                # for i_tau, current_tau in enumerate(np.linspace(512, 8192, 240)):
                for i_tau, current_tau in enumerate(np.linspace(8192, 32768, 240 * 2)):
                    vg_selected_gaussians = vg_layout_bundle.selectedGaussians(
                        camera=current_camera_instance_1080p,
                        tau=current_tau,
                    )
                    (
                        rendered_image_selected,
                        count_infrustum_vg,
                        _,
                        time_duration_vg_render,
                    ) = vg_selected_gaussians.renderReturnCountAndDuration(
                        camera=current_camera_instance_1080p
                    )

                    flip_values_vg: torch.Tensor = UTILITY_FLIP.CalculateFlipValues(
                        image_test=rendered_image_selected[:3]
                        * rendered_image_selected[3],
                        image_reference=rendered_image_gs_SSAA4K[:3]
                        * rendered_image_gs_SSAA4K[3],
                    )
                    flip_value_vg = flip_values_vg.mean().item()

                    UTILITY.SaveImage(
                        rendered_image_selected,
                        path=vg_select_config.OUTPUT_FOLDER_PATH
                        / f"images/vg_frame{i_tau}_tau{current_tau:.2f}_count{count_infrustum_vg}-{count_infrustum_vg/count_infrustum_gs_1080p*100:.2f}%-count{count_infrustum_gs_1080p}_flip{flip_value_vg:.4f}.png",
                    )

                    image_flip_vg = UTILITY_FLIP.FlipValuesToRgbHeatMap(flip_values_vg)
                    UTILITY.SaveImage(
                        image_flip_vg,
                        path=vg_select_config.OUTPUT_FOLDER_PATH
                        / f"images/flip_frame{i_tau}_tau{current_tau:.2f}_flip{flip_value_vg:.4f}.png",
                    )

                exit(0)

            if not vg_select_config.DISABLE_VG_AND_ONLY_RENDER_GS:
                for current_tau in vg_select_config.TAUS:
                    ExLog(f"{current_tau=}")

                    # [vg: for a given camera, render composed vg]

                    time_start_vg_select: float = time.perf_counter()
                    vg_selected_gaussians = vg_layout_bundle.selectedGaussians(
                        camera=current_camera_instance_1080p,
                        tau=current_tau,
                    )
                    time_end_vg_select: float = time.perf_counter()
                    time_duration_vg_select: float = (
                        time_end_vg_select - time_start_vg_select
                    )

                    # [vg: render composed gaussians]

                    (
                        rendered_image_selected,
                        count_infrustum_vg,
                        _,
                        time_duration_vg_render,
                    ) = vg_selected_gaussians.renderReturnCountAndDuration(
                        camera=current_camera_instance_1080p
                    )

                    if vg_select_config.CALCULATE_PSNR_WITH_NVS_GT:
                        psnr_value_vg_gt = UTILITY.Psnr(
                            rendered_image_selected[:3] * rendered_image_selected[3],
                            (
                                camera_gt_image[:3] * camera_gt_image[3]
                                if camera_gt_image.shape[0] == 4
                                else camera_gt_image
                            ),
                        ).item()
                        metrics_vg_psnr_nvs_gt.append(psnr_value_vg_gt)

                    flip_values_vg: torch.Tensor = UTILITY_FLIP.CalculateFlipValues(
                        image_test=rendered_image_selected[:3]
                        * rendered_image_selected[3],
                        image_reference=rendered_image_gs_SSAA4K[:3]
                        * rendered_image_gs_SSAA4K[3],
                    )
                    flip_value_vg = flip_values_vg.mean().item()
                    image_flip_vg = UTILITY_FLIP.FlipValuesToRgbHeatMap(flip_values_vg)
                    if vg_select_config.SAVE_IMAGES_COMPARISON:
                        UTILITY.SaveImage(
                            rendered_image_selected,
                            path=vg_select_config.OUTPUT_FOLDER_PATH
                            / f"images/{i_frame}_vg-tau{int(current_tau)}_count{count_infrustum_vg}-{count_infrustum_vg/count_infrustum_gs_1080p*100:.2f}%-count{count_infrustum_gs_1080p}_flip{flip_value_vg:.4f}_fps{1.0 / (time_duration_vg_select + time_duration_vg_render):.2f}-{(1.0 / (time_duration_vg_select + time_duration_vg_render))/(1.0 / time_duration_gs_render_1080p):.2f}x-fps{1.0 / time_duration_gs_render_1080p:.2f}.png",
                        )
                        UTILITY.SaveImage(
                            image_flip_vg,
                            path=vg_select_config.OUTPUT_FOLDER_PATH
                            / f"images/{i_frame}_vg-tau{int(current_tau)}-flip_flip{flip_value_vg:.4f}.png",
                        )
                    if vg_select_config.SAVE_IMAGES_CONTINUOUS:
                        UTILITY.SaveImage(
                            rendered_image_selected,
                            path=vg_select_config.OUTPUT_FOLDER_PATH
                            / f"images/vg-tau{int(current_tau)}_{i_frame}_count{count_infrustum_vg}-{count_infrustum_vg/count_infrustum_gs_1080p*100:.2f}%-count{count_infrustum_gs_1080p}_flip{flip_value_vg:.4f}_fps{1.0 / (time_duration_vg_select + time_duration_vg_render):.2f}-{(1.0 / (time_duration_vg_select + time_duration_vg_render))/(1.0 / time_duration_gs_render_1080p):.2f}x-fps{1.0 / time_duration_gs_render_1080p:.2f}.png",
                        )
                        UTILITY.SaveImage(
                            image_flip_vg,
                            path=vg_select_config.OUTPUT_FOLDER_PATH
                            / f"images/vg-tau{int(current_tau)}-flip_{i_frame}_flip{flip_value_vg:.4f}.png",
                        )

                    # [vg gsplat radius clip]

                    if vg_select_config.GSPLAT_RADIUS_CLIP:
                        for gsplat_radius_clip in [2.0]:
                            (
                                rendered_image_clip,
                                count_clip_infrustum,
                                _,
                                _,
                            ) = gs_layout_gaussians.renderReturnCountAndDuration(
                                camera=current_camera_instance_1080p,
                                gsplat_radius_clip=gsplat_radius_clip,
                            )
                            UTILITY.SaveImage(
                                rendered_image_clip,
                                path=vg_select_config.OUTPUT_FOLDER_PATH
                                / f"images/{i_frame}_clip{gsplat_radius_clip:.2f}_{count_clip_infrustum/count_infrustum_gs_1080p*100:.2f}%.png",
                            )

                            flip_values_vg: torch.Tensor = (
                                UTILITY_FLIP.CalculateFlipValues(
                                    image_test=rendered_image_clip[:3]
                                    * rendered_image_clip[3],
                                    image_reference=rendered_image_gs_SSAA4K[:3]
                                    * rendered_image_gs_SSAA4K[3],
                                )
                            )
                            flip_value_vg = flip_values_vg.mean().item()
                            image_flip_vg = UTILITY_FLIP.FlipValuesToRgbHeatMap(
                                flip_values_vg
                            )
                            UTILITY.SaveImage(
                                image_flip_vg,
                                path=vg_select_config.OUTPUT_FOLDER_PATH
                                / f"images/{i_frame}_flip_clip{gsplat_radius_clip:.2f}_{flip_value_vg:.4f}.png",
                            )

                    # [metrics]

                    metrics_gs_number_of_gaussians_in_frustum.append(
                        count_infrustum_gs_1080p
                    )
                    metrics_vg_number_of_gaussians_in_frustum.append(count_infrustum_vg)

                    metrics_gs_duration_all.append(time_duration_gs_render_1080p)
                    metrics_vg_duration_select.append(time_duration_vg_select)
                    metrics_vg_duration_render.append(time_duration_vg_render)
                    metrics_vg_duration_all.append(
                        time_duration_vg_select + time_duration_vg_render
                    )

                    metrics_gs_fps.append(1.0 / time_duration_gs_render_1080p)
                    metrics_vg_fps.append(
                        1.0 / (time_duration_vg_select + time_duration_vg_render)
                    )

                    metrics_gs_over_gs_SSAA4K_flip.append(flip_value_gs_1080p)
                    metrics_vg_over_gs_SSAA4K_flip.append(flip_value_vg)

        i_frame_offset += len(current_cameras)

    # [metrics for the first layout]
    count_frames = len(metrics_gs_number_of_gaussians_in_frustum)
    metrics_content = f"{count_frames=}\n\n"
    # metrics: all frames
    metrics_content += f"{metrics_gs_number_of_gaussians_in_frustum=}\n{metrics_vg_number_of_gaussians_in_frustum=}\n\n{metrics_gs_duration_all=}\n{metrics_vg_duration_select=}\n{metrics_vg_duration_render=}\n{metrics_vg_duration_all=}\n\n{metrics_gs_fps=}\n{metrics_vg_fps=}\n\n{metrics_gs_over_gs_SSAA4K_flip=}\n{metrics_vg_over_gs_SSAA4K_flip=}\n\n"
    if vg_select_config.CALCULATE_PSNR_WITH_NVS_GT:
        metrics_content += f"{metrics_gs_psnr_nvs_gt=}\n\n"
        metrics_content += f"{metrics_vg_psnr_nvs_gt=}\n\n"
    # metrics: average on all frames
    avg_metrics_gs_number_of_gaussians_in_frustum: float = (
        sum(metrics_gs_number_of_gaussians_in_frustum) / count_frames
    )
    avg_metrics_vg_number_of_gaussians_in_frustum: float = (
        sum(metrics_vg_number_of_gaussians_in_frustum) / count_frames
    )
    avg_metrics_gs_duration_all: float = sum(metrics_gs_duration_all) / count_frames
    avg_metrics_vg_duration_select: float = (
        sum(metrics_vg_duration_select) / count_frames
    )
    avg_metrics_vg_duration_render: float = (
        sum(metrics_vg_duration_render) / count_frames
    )
    avg_metrics_vg_duration_all: float = sum(metrics_vg_duration_all) / count_frames
    avg_metrics_gs_fps: float = sum(metrics_gs_fps) / count_frames
    avg_metrics_vg_fps: float = sum(metrics_vg_fps) / count_frames
    avg_metrics_gs_flip: float = sum(metrics_gs_over_gs_SSAA4K_flip) / count_frames
    avg_metrics_vg_flip: float = sum(metrics_vg_over_gs_SSAA4K_flip) / count_frames
    metrics_content += f"{avg_metrics_gs_number_of_gaussians_in_frustum=}\n{avg_metrics_vg_number_of_gaussians_in_frustum=}\n\n{avg_metrics_gs_duration_all=}\n{avg_metrics_vg_duration_select=}\n{avg_metrics_vg_duration_render=}\n{avg_metrics_vg_duration_all=}\n\n{avg_metrics_gs_fps=}\n{avg_metrics_vg_fps=}\n\n{avg_metrics_gs_flip=}\n{avg_metrics_vg_flip=}\n\n"
    if vg_select_config.CALCULATE_PSNR_WITH_NVS_GT:
        avg_metrics_psnr_nvs_gs_gt: float = sum(metrics_gs_psnr_nvs_gt) / count_frames
        metrics_content += f"{avg_metrics_psnr_nvs_gs_gt=}\n"
        avg_metrics_psnr_nvs_vg_gt: float = sum(metrics_vg_psnr_nvs_gt) / count_frames
        metrics_content += f"{avg_metrics_psnr_nvs_vg_gt=}\n\n"

    # metrics: average on each distance
    if vg_select_config.SAVE_METRICS_AT_DIFFERENT_DISTANCES:
        assert count_frames == 400

        distances_avg_metrics_gs_number_of_gaussians_in_frustum: list[float] = [
            0.0
        ] * 20
        distances_avg_metrics_vg_number_of_gaussians_in_frustum: list[float] = [
            0.0
        ] * 20
        distances_avg_metrics_gs_duration_all: list[float] = [0.0] * 20
        distances_avg_metrics_vg_duration_select: list[float] = [0.0] * 20
        distances_avg_metrics_vg_duration_render: list[float] = [0.0] * 20
        distances_avg_metrics_vg_duration_all: list[float] = [0.0] * 20
        distances_avg_metrics_gs_fps: list[float] = [0.0] * 20
        distances_avg_metrics_vg_fps: list[float] = [0.0] * 20
        distances_avg_metrics_gs_flip: list[float] = [0.0] * 20
        distances_avg_metrics_vg_flip: list[float] = [0.0] * 20

        for i_frame in range(count_frames):
            distances_avg_metrics_gs_number_of_gaussians_in_frustum[
                i_frame % 20
            ] += metrics_gs_number_of_gaussians_in_frustum[i_frame]
            distances_avg_metrics_vg_number_of_gaussians_in_frustum[
                i_frame % 20
            ] += metrics_vg_number_of_gaussians_in_frustum[i_frame]
            distances_avg_metrics_gs_duration_all[
                i_frame % 20
            ] += metrics_gs_duration_all[i_frame]
            distances_avg_metrics_vg_duration_select[
                i_frame % 20
            ] += metrics_vg_duration_select[i_frame]
            distances_avg_metrics_vg_duration_render[
                i_frame % 20
            ] += metrics_vg_duration_render[i_frame]
            distances_avg_metrics_vg_duration_all[
                i_frame % 20
            ] += metrics_vg_duration_all[i_frame]
            distances_avg_metrics_gs_fps[i_frame % 20] += metrics_gs_fps[i_frame]
            distances_avg_metrics_vg_fps[i_frame % 20] += metrics_vg_fps[i_frame]
            distances_avg_metrics_gs_flip[
                i_frame % 20
            ] += metrics_gs_over_gs_SSAA4K_flip[i_frame]
            distances_avg_metrics_vg_flip[
                i_frame % 20
            ] += metrics_vg_over_gs_SSAA4K_flip[i_frame]
        for i_distance in range(20):
            distances_avg_metrics_gs_number_of_gaussians_in_frustum[i_distance] /= 20
            distances_avg_metrics_vg_number_of_gaussians_in_frustum[i_distance] /= 20
            distances_avg_metrics_gs_duration_all[i_distance] /= 20
            distances_avg_metrics_vg_duration_select[i_distance] /= 20
            distances_avg_metrics_vg_duration_render[i_distance] /= 20
            distances_avg_metrics_vg_duration_all[i_distance] /= 20
            distances_avg_metrics_gs_fps[i_distance] /= 20
            distances_avg_metrics_vg_fps[i_distance] /= 20
            distances_avg_metrics_gs_flip[i_distance] /= 20
            distances_avg_metrics_vg_flip[i_distance] /= 20

        metrics_distance: list[float] = list(range(20))
        metrics_content += f"{metrics_distance=}\n\n"
        metrics_content += f"{distances_avg_metrics_gs_number_of_gaussians_in_frustum=}\n{distances_avg_metrics_vg_number_of_gaussians_in_frustum=}\n\n{distances_avg_metrics_gs_duration_all=}\n{distances_avg_metrics_vg_duration_select=}\n{distances_avg_metrics_vg_duration_render=}\n{distances_avg_metrics_vg_duration_all=}\n\n{distances_avg_metrics_gs_fps=}\n{distances_avg_metrics_vg_fps=}\n\n{distances_avg_metrics_gs_flip=}\n{distances_avg_metrics_vg_flip=}\n\n"

    with open(vg_select_config.OUTPUT_FOLDER_PATH / "_metrics.py", "w") as f:
        f.write(metrics_content)


if __name__ == "__main__":
    ExLog("START")
    print()

    setup_torch_and_random()

    parser = argparse.ArgumentParser()
    vg_select_config = VGSelectConfig(parser=parser)
    args = parser.parse_args()
    vg_select_config.extract(args=args)
    vg_select_config.process()

    with ExTimer("VgSelect()"):
        VgSelect(vg_select_config=vg_select_config)

    print()
    ExLog("END")
