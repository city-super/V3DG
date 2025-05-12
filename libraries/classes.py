# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import pathlib, math, time, functools, datetime, zoneinfo

import numpy as np
import plyfile
import torch
import einops

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
import scipy.spatial.transform

# https://tqdm.github.io/docs/tqdm/#tqdm-objects
import tqdm

# https://docs.gsplat.studio/main/index.html
import gsplat

from libraries.utilities import ExLog, ExTimer, UTILITY
from libraries.cliconfigs import VGBuildConfig


class Camera:
    def DeriveRandomPoseLookingAtOrigin(
        center: torch.Tensor,
        radius: float,
        asset_name: str = "donut",
    ) -> Camera:
        """
        center: (3,)
        radius: pass in four times of the furthest distance in a cluster / cluster group
        """
        camera_position_z = -radius + 2.0 * radius * torch.rand(1)
        camera_position_xy_radius = torch.sqrt(radius**2 - camera_position_z**2)
        camera_position_xy_theta = torch.rand(1) * 2.0 * torch.pi
        camera_position_x = camera_position_xy_radius * torch.cos(
            camera_position_xy_theta
        )
        camera_position_y = camera_position_xy_radius * torch.sin(
            camera_position_xy_theta
        )
        camera_position_relative = torch.tensor(
            [camera_position_x, camera_position_y, camera_position_z]
        )
        camera_position_absolute = center + camera_position_relative

        camera_lookat = (
            -camera_position_relative / camera_position_relative.pow(2).sum().sqrt()
        )
        camera_upward = torch.tensor([camera_lookat[1], -camera_lookat[0], 0])
        camera_upward = camera_upward / camera_upward.pow(2).sum().sqrt()
        camera_cross = torch.cross(camera_lookat, camera_upward, dim=0)

        # use row vector
        T = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [
                    -camera_position_absolute[0],
                    -camera_position_absolute[1],
                    -camera_position_absolute[2],
                    1,
                ],
            ]
        )
        R = torch.tensor(
            [
                [camera_cross[0], -camera_upward[0], camera_lookat[0], 0],
                [camera_cross[1], -camera_upward[1], camera_lookat[1], 0],
                [camera_cross[2], -camera_upward[2], camera_lookat[2], 0],
                [0, 0, 0, 1],
            ]
        )

        if asset_name == "donut":
            original_image_width = 800
            original_image_height = 800
            original_focal_x = 1111.1111350971692
            original_focal_y = 1111.1111350971692
        else:
            raise NotImplementedError

        # TODO This can be modified.
        image_width = 64
        image_height = 64
        # TODO keep fov unchanged?
        focal_x = original_focal_x * image_width / original_image_width
        focal_y = original_focal_y * image_height / original_image_height

        camera_temp = Camera(
            image_width=image_width,
            image_height=image_height,
            focal_x=focal_x,
            focal_y=focal_y,
            # don't use the initialization of view_matrix
            R=torch.eye(3),
            t=torch.tensor(
                [0.0, 0.0, 0.0],
                dtype=torch.float,
            ),
        )
        camera_temp.view_matrix = T @ R

        return camera_temp

    def DeriveSixDirections(
        center: torch.Tensor,
        distance: float,
        asset_name: str = "donut",
    ) -> list[Camera]:
        # TODO explore this! how to give the parameters
        if asset_name == "donut":
            original_image_width = 800
            original_image_height = 800
            original_focal_x = 1111.1111350971692
            original_focal_y = 1111.1111350971692
        else:
            raise NotImplementedError

        # (1, 3) -> (3,)
        center = center[0]

        # TODO This can be modified.
        image_width = 64
        image_height = 64
        # TODO keep fov unchanged?
        focal_x = original_focal_x * image_width / original_image_width
        focal_y = original_focal_y * image_height / original_image_height

        # front, left, right, back, up, down
        R_front = torch.tensor(
            scipy.spatial.transform.Rotation.from_rotvec(
                [0, 0, 180], degrees=True
            ).as_matrix(),
            dtype=torch.float,
        ) @ torch.tensor(
            scipy.spatial.transform.Rotation.from_rotvec(
                [-90, 0, 0], degrees=True
            ).as_matrix(),
            dtype=torch.float,
        )
        cameras: list[Camera] = [
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=R_front,
                t=torch.tensor(
                    [center[0], center[1] + distance, center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [0, 0, -90], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0] + distance, center[1], center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [0, 0, 90], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0] - distance, center[1], center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [0, 0, 180], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0], center[1] - distance, center[2]], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [90, 0, 0], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0], center[1], center[2] + distance], dtype=torch.float
                ),
            ),
            Camera(
                image_width=image_width,
                image_height=image_height,
                focal_x=focal_x,
                focal_y=focal_y,
                R=torch.tensor(
                    scipy.spatial.transform.Rotation.from_rotvec(
                        [-90, 0, 0], degrees=True
                    ).as_matrix(),
                    dtype=torch.float,
                )
                @ R_front,
                t=torch.tensor(
                    [center[0], center[1], center[2] - distance], dtype=torch.float
                ),
            ),
        ]

        return cameras

    def FocalToFov(focal, pixels):
        return 2 * math.atan(pixels / (2 * focal))

    def GetProjectionMatrix(z_near, z_far, fov_x, fov_y):
        tan_half_fov_x = math.tan((fov_x / 2))
        tan_half_fov_y = math.tan((fov_y / 2))

        top = tan_half_fov_y * z_near
        bottom = -top
        right = tan_half_fov_x * z_near
        left = -right

        P = torch.zeros(4, 4)

        z_sign = 1.0

        P[0, 0] = 2.0 * z_near / (right - left)
        P[1, 1] = 2.0 * z_near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * z_far / (z_far - z_near)
        P[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return P

    def __init__(self, image_width, image_height, focal_x, focal_y, R, t) -> None:
        self.image_width = image_width
        self.image_height = image_height
        self.focal_x = focal_x
        self.focal_y = focal_y

        self.fov_x = Camera.FocalToFov(focal=self.focal_x, pixels=self.image_width)
        self.fov_y = Camera.FocalToFov(focal=self.focal_y, pixels=self.image_height)

        # [view matrix]

        Rt = torch.zeros((4, 4), dtype=torch.float)
        Rt[:3, :3] = R
        Rt[:3, 3] = t
        Rt[3, 3] = 1.0
        Rt_inv: torch.Tensor = torch.linalg.inv(Rt)
        # homogenous coordinate, row vector
        self.view_matrix = Rt_inv.transpose(0, 1)

        # [projection matrix]

        z_far = 100.0  # no use
        z_near = 0.01  # no use
        # homogenous coordinate, row vector
        self.projection_matrix = Camera.GetProjectionMatrix(
            z_near=z_near, z_far=z_far, fov_x=self.fov_x, fov_y=self.fov_y
        ).transpose(0, 1)


class LearnableGaussians(torch.nn.Module):
    def ActivationScales(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def InverseActivationScales(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y)

    def ActivationQauternions(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x)

    def ActivationOpacities(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def InverseActivationOpacities(y: torch.Tensor) -> torch.Tensor:
        return torch.log(y / (1 - y))

    def ActivationSh0(x: torch.Tensor) -> torch.Tensor:
        SH_C0 = 0.28209479177387814
        return torch.clip(x * SH_C0 + 0.5, min=0.0, max=None)

    def InverseActivationSh0(y: torch.Tensor) -> torch.Tensor:
        SH_C0 = 0.28209479177387814
        return (y - 0.5) / SH_C0

    def __init__(
        self,
        vg_build_config: VGBuildConfig,
        original_gaussians: Cluster,
        gaussians: Cluster,
        center: torch.Tensor,
        distance: float,
    ) -> None:
        """
        pass in initialized 3D Gaussians. LearnableGaussians is only responsible for the optimization task. The count of 3D Gaussians won't change during optimization.
        """
        super().__init__()

        self.vg_build_config = vg_build_config

        self.count = gaussians.count

        self.center = center
        self.distance = distance

        self.original_gaussians = original_gaussians

        if gaussians.positions.isnan().any():
            ExLog("LearnableGaussian.__init__() has nan!!!", "ERROR")
            exit(-1)

        with torch.no_grad():
            self.parameters_positions: torch.Tensor = torch.nn.Parameter(
                gaussians.positions.clone(), requires_grad=True
            )
            self.parameters_scales: torch.Tensor = torch.nn.Parameter(
                LearnableGaussians.InverseActivationScales(gaussians.scales.clone()),
                requires_grad=True,
            )
            self.parameters_quaternions: torch.Tensor = torch.nn.Parameter(
                gaussians.quaternions.clone(), requires_grad=True
            )
            self.parameters_opacities: torch.Tensor = torch.nn.Parameter(
                LearnableGaussians.InverseActivationOpacities(
                    gaussians.opacities.clone()
                ),
                requires_grad=True,
            )
            self.parameters_sh0: torch.Tensor = torch.nn.Parameter(
                LearnableGaussians.InverseActivationSh0(gaussians.rgbs.clone()),
                requires_grad=True,
            )

    @property
    def positions(self) -> torch.Tensor:
        return self.parameters_positions

    @property
    def scales(self) -> torch.Tensor:
        return LearnableGaussians.ActivationScales(self.parameters_scales)

    @property
    def quaternions(self) -> torch.Tensor:
        return LearnableGaussians.ActivationQauternions(self.parameters_quaternions)

    @property
    def opacities(self) -> torch.Tensor:
        return LearnableGaussians.ActivationOpacities(self.parameters_opacities)

    @property
    def rgbs(self) -> torch.Tensor:
        return LearnableGaussians.ActivationSh0(self.parameters_sh0)

    # NOTICE: Directly call functions in class Cluster.

    def render(self, *args, **kwargs):
        return Cluster.render(self, *args, **kwargs)

    def renderReturnCountAndDuration(self, *args, **kwargs):
        return Cluster.renderReturnCountAndDuration(self, *args, **kwargs)

    def renderFullImageConsolidatingSixDirections(self, *args, **kwargs):
        return Cluster.renderFullImageConsolidatingSixDirections(self, *args, **kwargs)

    def train(self) -> None:
        # https://pytorch.org/docs/stable/optim.html
        parameters = [
            {
                "params": [self.parameters_positions],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_POSITION,
            },
            {
                "params": [self.parameters_scales],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_SCALE,
            },
            {
                "params": [self.parameters_quaternions],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_QUATERNION,
            },
            {
                "params": [self.parameters_opacities],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_OPACITY,
            },
            {
                "params": [self.parameters_sh0],
                "lr": self.vg_build_config.SIMPLIFICATION_LEARNING_RATE_SH0,
            },
        ]
        optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

        # DEBUG save intermediate results
        current_time_str = datetime.datetime.now(
            tz=zoneinfo.ZoneInfo("Asia/Shanghai")
        ).strftime("%y%m%d-%H%M%S")
        if self.vg_build_config.SAVE_IMAGES_DURING_OPTIMIZATION:
            UTILITY.SaveImage(
                self.original_gaussians.renderFullImageConsolidatingSixDirections(
                    center=self.center, distance=self.distance
                ),
                self.vg_build_config.OUTPUT_FOLDER_PATH
                / f"images/{current_time_str}-original.png",
            )
        for iter in range(self.vg_build_config.SIMPLIFICATION_ITERATION + 1):
            # [calculate loss]

            random_camera_looking_at_center = Camera.DeriveRandomPoseLookingAtOrigin(
                center=self.center[0], radius=self.distance
            )

            # (4, h, w)
            image_gt: torch.Tensor = self.original_gaussians.render(
                camera=random_camera_looking_at_center
            )
            image_render: torch.Tensor = self.render(
                camera=random_camera_looking_at_center
            )

            if self.vg_build_config.SAVE_IMAGES_DURING_OPTIMIZATION:
                if iter % 160 == 0:
                    UTILITY.SaveImage(
                        self.renderFullImageConsolidatingSixDirections(
                            center=self.center, distance=self.distance
                        ),
                        self.vg_build_config.OUTPUT_FOLDER_PATH
                        / f"images/{current_time_str}-iter{iter}.png",
                    )

            # add black background
            loss_l1: torch.Tensor = UTILITY.L1Loss(
                image=image_render[:3] * image_render[3],
                target=image_gt[:3] * image_gt[3],
            )
            ssim: torch.Tensor = UTILITY.Ssim(
                image=image_render[:3] * image_render[3],
                target=image_gt[:3] * image_gt[3],
            )

            loss_dssim: torch.Tensor = 1.0 - ssim
            # add alpha channel supervision here
            loss: torch.Tensor = (
                (1.0 - self.vg_build_config.SIMPLIFICATION_LOSS_LAMBDA_DSSIM) * loss_l1
                + self.vg_build_config.SIMPLIFICATION_LOSS_LAMBDA_DSSIM * loss_dssim
                + 0.1 * torch.abs((image_render[3] - image_gt[3])).mean()
            )

            # [backward]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def toGaussians(self, lod_level: int) -> Cluster:
        return Cluster(
            vg_build_config=self.vg_build_config,
            count=self.count,
            positions=self.positions.clone().detach().requires_grad_(False),
            scales=self.scales.clone().detach().requires_grad_(False),
            quaternions=self.quaternions.clone().detach().requires_grad_(False),
            opacities=self.opacities.clone().detach().requires_grad_(False),
            rgbs=self.rgbs.clone().detach().requires_grad_(False),
            lod_level=lod_level,
        )


class ClustersList:
    def __init__(
        self,
        vg_build_config: VGBuildConfig,
        clusters_list: list[Clusters],
    ) -> None:
        self.vg_build_config = vg_build_config

        self.clusters_list: list[Clusters] = clusters_list
        self.count: int = len(self.clusters_list)

    def append(self, clusters: Clusters) -> None:
        self.clusters_list.append(clusters)
        self.count = len(self.clusters_list)

    def extend(self, clusters_list: list[Clusters]) -> None:
        self.clusters_list.extend(clusters_list)
        self.count = len(self.clusters_list)

    def consolidateIntoClusters(self) -> Clusters:
        clusters: list[Cluster] = functools.reduce(
            lambda a, b: a + b.clusters, self.clusters_list, []
        )
        return Clusters(
            vg_build_config=self.vg_build_config, clusters=clusters, lod_level=None
        )

    def savePlyWithDifferentColors(self, path: pathlib.Path) -> None:
        color_choices = np.random.randint(
            low=0, high=255, size=(self.count, 3), dtype=np.uint8
        )

        ply_points = np.concatenate(
            [
                np.concatenate(
                    [
                        clusters.consolidateIntoASingleCluster()
                        .positions.cpu()
                        .numpy(),
                        np.zeros(
                            clusters.consolidateIntoASingleCluster().positions.shape,
                            dtype=np.uint8,
                        )
                        + color_choices[i],
                    ],
                    axis=1,
                )
                for i, clusters in enumerate(self.clusters_list)
            ],
            axis=0,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )

        ExLog(f"Save {self.count} cluster groups at {path}.")

    def saveBundle(self) -> None:
        # [save clusters.npz]

        clusters_count = 0
        for clusters in self.clusters_list:
            clusters_count += clusters.count
            ExLog(
                f"LOD{clusters.lod_level}, {clusters.count} clusters, gaussians in each cluster {[cluster.count for cluster in clusters.clusters]}",
                "DEBUG",
            )

        lod_levels = np.zeros((clusters_count, 1), dtype=np.int32)
        start_indices = np.zeros((clusters_count, 1), dtype=np.int32)
        counts = np.zeros((clusters_count, 1), dtype=np.int32)
        child_centers = np.zeros((clusters_count, 3), dtype=np.float32)
        parent_centers = np.zeros((clusters_count, 3), dtype=np.float32)
        child_radii = np.zeros((clusters_count, 1), dtype=np.float32)
        parent_radii = np.zeros((clusters_count, 1), dtype=np.float32)

        start_index = 0
        i_cluster = 0
        for clusters in self.clusters_list:
            for cluster in clusters.clusters:
                lod_levels[i_cluster] = cluster.lod_level
                start_indices[i_cluster] = start_index
                counts[i_cluster] = cluster.count
                child_centers[i_cluster] = cluster.child_center_in_cluster_group
                parent_centers[i_cluster] = cluster.parent_center_in_cluster_group
                child_radii[i_cluster] = cluster.child_radius_in_cluster_group
                parent_radii[i_cluster] = cluster.parent_radius_in_cluster_group

                start_index += cluster.count
                i_cluster += 1

        np.savez(
            self.vg_build_config.BUNDLE_CLUSTERS_NPZ_PATH,
            lod_levels=lod_levels,
            start_indices=start_indices,
            counts=counts,
            child_centers=child_centers,
            parent_centers=parent_centers,
            child_radii=child_radii,
            parent_radii=parent_radii,
        )

        # [save gaussians.npz]

        np.savez(
            self.vg_build_config.BUNDLE_GAUSSIANS_NPZ_PATH,
            positions=np.concatenate(
                [
                    cluster.positions.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            scales=np.concatenate(
                [
                    cluster.scales.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            quaternions=np.concatenate(
                [
                    cluster.quaternions.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            opacities=np.concatenate(
                [
                    cluster.opacities.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
            rgbs=np.concatenate(
                [
                    cluster.rgbs.cpu().numpy()
                    for cluster in self.consolidateIntoClusters().clusters
                ],
                axis=0,
            ),
        )


class Clusters:
    def __init__(
        self,
        clusters: list[Cluster],
        vg_build_config: VGBuildConfig = None,
        lod_level: int | None = None,
    ) -> None:
        """
        Assign `lod_level` an int if all clusters are at the same lod level.
        """
        self.vg_build_config = vg_build_config

        self.clusters = clusters
        self.count = len(self.clusters)
        self.lod_level = lod_level

    def updatePositionsOfAllClusters(self) -> None:
        # 241105 change property of positions to instance variable
        positions_of_all_clusters = torch.zeros((self.count, 3), dtype=torch.float32)
        for i_cluster in range(self.count):
            positions_of_all_clusters[i_cluster] = self.clusters[i_cluster].getCenter()
        self.positions = positions_of_all_clusters

    def append(self, cluster: Cluster) -> None:
        self.clusters.append(cluster)
        self.count = len(self.clusters)

    def extend(self, clusters: list[Cluster]) -> None:
        self.clusters.extend(clusters)
        self.count = len(self.clusters)

    def consolidateIntoASingleCluster(self) -> Cluster:
        count_simplified = sum([c.count for c in self.clusters])
        positions_simplified = torch.cat([c.positions for c in self.clusters], dim=0)
        scales_simplified = torch.cat([c.scales for c in self.clusters], dim=0)
        quaternions_simplified = torch.cat(
            [c.quaternions for c in self.clusters], dim=0
        )
        opacities_simplified = torch.cat([c.opacities for c in self.clusters], dim=0)
        rgbs_simplified = torch.cat([c.rgbs for c in self.clusters], dim=0)
        return Cluster(
            vg_build_config=self.vg_build_config,
            count=count_simplified,
            positions=positions_simplified,
            scales=scales_simplified,
            quaternions=quaternions_simplified,
            opacities=opacities_simplified,
            rgbs=rgbs_simplified,
            lod_level=self.lod_level,
        )

    def splitIntoClusterGroups(self) -> ClustersList:
        # # [clusters -> cluster groups]

        # count_cluster_groups = int(
        #     self.count
        #     / self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
        # )

        # if count_cluster_groups >= 2:
        #     cluster_centers = np.zeros((self.count, 3))
        #     for i_cluster_group in range(self.count):
        #         cluster_centers[i_cluster_group] = (
        #             self.clusters[i_cluster_group].positions.mean(dim=0).cpu()
        #         )

        #     with ExTimer("kmeans"):
        #         kmeans = sklearn.cluster.MiniBatchKMeans(
        #             n_clusters=count_cluster_groups,
        #             init="k-means++",
        #             n_init="auto",
        #             random_state=0,
        #         ).fit(cluster_centers)

        #     labels = torch.from_numpy(kmeans.labels_)

        #     cluster_groups: ClustersList = ClustersList(
        #         vg_build_config=self.vg_build_config, clusters_list=[]
        #     )
        #     with ExTimer("form CG"):
        #         # find clusters in current cluster group
        #         for i_cluster_group in range(count_cluster_groups):
        #             cluster_group: Clusters = Clusters(
        #                 vg_build_config=self.vg_build_config,
        #                 clusters=[
        #                     self.clusters[c]
        #                     for c in torch.where(labels == i_cluster_group)[0]
        #                 ],
        #                 lod_level=self.lod_level,
        #             )
        #             ExLog(f"{i_cluster_group=} {cluster_group.count=} clusters.counts={[cluster.count for cluster in cluster_group.clusters]}", "DEBUG")
        #             if cluster_group.count >= 2:
        #                 cluster_groups.append(cluster_group)
        # else:
        #     # only one cluster group
        #     cluster_groups: ClustersList = ClustersList(
        #         vg_build_config=self.vg_build_config, clusters_list=[self]
        #     )

        # cluster_groups.savePlyWithDifferentColors(
        #     path=self.vg_build_config.OUTPUT_FOLDER_PATH
        #     / f"plys/lod{self.lod_level}-to-lod{self.lod_level+1}-cluster-groups.ply"
        # )

        # [new version: replace kmeans with median split]

        if (
            self.count
            > self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
        ):
            all_complete_cluster_groups: list[Clusters] = []
            all_incomplete_cluster_groups: list[Clusters] = [self]

            while len(all_incomplete_cluster_groups) != 0:
                current_incomplete_cluster_groups: list[Clusters] = []
                for incomplete_cluster_group in all_incomplete_cluster_groups:
                    lengths = torch.tensor(
                        [
                            incomplete_cluster_group.positions[:, 0].max().item()
                            - incomplete_cluster_group.positions[:, 0].min().item(),
                            incomplete_cluster_group.positions[:, 1].max().item()
                            - incomplete_cluster_group.positions[:, 1].min().item(),
                            incomplete_cluster_group.positions[:, 2].max().item()
                            - incomplete_cluster_group.positions[:, 2].min().item(),
                        ]
                    )
                    axis_to_split = lengths.argmax().item()
                    axis_median = (
                        incomplete_cluster_group.positions[:, axis_to_split]
                        .median()
                        .item()
                    )

                    cluster_group_left: Clusters = Clusters(
                        clusters=[],
                        vg_build_config=self.vg_build_config,
                        lod_level=self.lod_level,
                    )
                    cluster_group_right: Clusters = Clusters(
                        clusters=[],
                        vg_build_config=self.vg_build_config,
                        lod_level=self.lod_level,
                    )
                    for i_cluster in range(incomplete_cluster_group.count):
                        if (
                            incomplete_cluster_group.positions[i_cluster, axis_to_split]
                            <= axis_median
                        ):
                            cluster_group_left.append(
                                incomplete_cluster_group.clusters[i_cluster]
                            )
                        else:
                            cluster_group_right.append(
                                incomplete_cluster_group.clusters[i_cluster]
                            )

                    # update positions
                    cluster_group_left.positions = incomplete_cluster_group.positions[
                        incomplete_cluster_group.positions[:, axis_to_split]
                        <= axis_median
                    ]
                    cluster_group_right.positions = incomplete_cluster_group.positions[
                        incomplete_cluster_group.positions[:, axis_to_split]
                        > axis_median
                    ]
                    # ExLog(
                    #     f"{cluster_group_left.positions.shape=} {cluster_group_right.positions.shape=}",
                    #     "DEBUG",
                    # )

                    if (
                        cluster_group_left.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
                    ):
                        all_complete_cluster_groups.append(cluster_group_left)
                    else:
                        current_incomplete_cluster_groups.append(cluster_group_left)

                    if (
                        cluster_group_right.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_CLUSTERS_IN_ONE_CLUSTER_GROUP
                    ):
                        all_complete_cluster_groups.append(cluster_group_right)
                    else:
                        current_incomplete_cluster_groups.append(cluster_group_right)

                    # ExLog(
                    #     f"{len(all_complete_clusters)=} {len(all_incomplete_clusters)=} {len(current_incomplete_clusters)=} {incomplete_cluster.count=} {cluster_left.count=} {cluster_right.count=}"
                    # )

                all_incomplete_cluster_groups = current_incomplete_cluster_groups

            cluster_groups: ClustersList = ClustersList(
                vg_build_config=self.vg_build_config,
                clusters_list=all_complete_cluster_groups,
            )
        else:
            # only one cluster group
            cluster_groups: ClustersList = ClustersList(
                vg_build_config=self.vg_build_config,
                clusters_list=[self],
            )

        cluster_groups.savePlyWithDifferentColors(
            path=self.vg_build_config.OUTPUT_FOLDER_PATH
            / f"plys/lod{self.lod_level}-to-lod{self.lod_level+1}-cluster-groups.ply"
        )

        # ExLog(f"There are {cluster_groups.count} cluster groups.", "DEBUG")
        # for i_cluster_group in range(cluster_groups.count):
        #     current_cluster_group = cluster_groups.clusters_list[i_cluster_group]
        # ExLog(
        #     f"CG_{i_cluster_group}: {current_cluster_group.count} clusters; gaussians in clusters={[cluster.count for cluster in current_cluster_group.clusters]}",
        #     "DEBUG",
        # )

        return cluster_groups

    def setParentCenterAndRadiusValueForFinerLodLayerInClusterGroup(
        self, center: list[float], radius_value: float
    ) -> None:
        for cluster in self.clusters:
            cluster.parent_radius_in_cluster_group = radius_value
            cluster.parent_center_in_cluster_group = center

    def setChildCenterAndRadiusValueForCoarserLodLayerInClusterGroup(
        self, center: list[float], radius_value: float
    ) -> None:
        for cluster in self.clusters:
            cluster.child_radius_in_cluster_group = radius_value
            cluster.child_center_in_cluster_group = center

    def buildCoarserLodLayer(self) -> Clusters:
        """
        For all clusters in current lod layer, first form cluster groups.
        Then for each cluster group, merge, simplify, split to derive coarser clusters.
        Finally consolidate all coarser clusters to form the coarser lod layer.
        """

        current_lod_layer = self

        cluster_groups = current_lod_layer.splitIntoClusterGroups()
        ExLog(
            f"Cluster Group: {current_lod_layer.count} clusters -> {cluster_groups.count} cluster groups.",
        )

        coarser_lod_layer: Clusters = Clusters(
            vg_build_config=self.vg_build_config,
            clusters=[],
            lod_level=self.lod_level + 1,
        )
        ExLog(f"For each cluster group, execute Merge(), Simplify(), Split()...")
        for original_clusters_in_cluster_group in tqdm.tqdm(
            cluster_groups.clusters_list,
            bar_format=r"simplify cluster groups |{bar}| {n_fmt}/{total_fmt} {rate_fmt} {elapsed} ",
        ):
            # [merge]

            original_gaussians_in_cluster_group: Cluster = (
                original_clusters_in_cluster_group.consolidateIntoASingleCluster()
            )

            # [simplify]

            simplified_gaussians_in_cluster_group = (
                original_gaussians_in_cluster_group.downSampleHalf()
            )

            if self.vg_build_config.SIMPLIFICATION_ITERATION != 0:
                simplified_gaussians_in_cluster_group = (
                    simplified_gaussians_in_cluster_group.optimizeUsingLocalSplatting(
                        original_gaussians_in_cluster_group
                    )
                )

            if simplified_gaussians_in_cluster_group.count == 0:
                ExLog(f"[ERROR]", f"{original_gaussians_in_cluster_group.count=}")
                exit(-1)

            # [split]

            simplified_clusters_in_cluster_group = (
                simplified_gaussians_in_cluster_group.splitIntoClusters()
            )

            # [calculate and set value for selection]

            # 240809-0910: use original_gaussians instead of simplified_gaussians
            cluster_group_center = original_gaussians_in_cluster_group.getCenter()
            # 240809-0910: use original_gaussians instead of simplified_gaussians
            # ensure monotonic for selection in parallel
            cluster_group_radius_value = max(
                original_gaussians_in_cluster_group.getRadiusValueInClusterGroupForSelection(),
                max(
                    [
                        cluster.child_radius_in_cluster_group
                        for cluster in original_clusters_in_cluster_group.clusters
                    ]
                ),
            )

            original_clusters_in_cluster_group.setParentCenterAndRadiusValueForFinerLodLayerInClusterGroup(
                center=cluster_group_center.reshape((3,)).tolist(),
                radius_value=cluster_group_radius_value,
            )
            simplified_clusters_in_cluster_group.setChildCenterAndRadiusValueForCoarserLodLayerInClusterGroup(
                center=cluster_group_center.reshape((3,)).tolist(),
                radius_value=cluster_group_radius_value,
            )

            # [collect clusters in coarser lod layer]

            coarser_lod_layer.extend(simplified_clusters_in_cluster_group.clusters)

        return coarser_lod_layer

    def savePlyWithDifferentColors(
        self,
        path: pathlib.Path,
    ) -> None:
        color_choices = np.random.randint(
            low=0, high=255, size=(self.count, 3), dtype=np.uint8
        )

        ply_points = np.concatenate(
            [
                np.concatenate(
                    [
                        cluster.positions.cpu().numpy(),
                        np.zeros(cluster.positions.shape, dtype=np.uint8)
                        + color_choices[i],
                    ],
                    axis=1,
                )
                for i, cluster in enumerate(self.clusters)
            ],
            axis=0,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )

        ExLog(f"Save ply with {self.count} clusters at {path}.")

    def saveUsefulImages(self) -> None:
        consolited_cluster_of_current_layer = self.consolidateIntoASingleCluster()
        center = consolited_cluster_of_current_layer.getCenter()
        radius = consolited_cluster_of_current_layer.getFarthestDistanceInClusterGroup()
        lod0_image_full = consolited_cluster_of_current_layer.renderFullImageConsolidatingSixDirections(
            center=center,
            distance=radius * 4,
        )
        UTILITY.SaveImage(
            image=lod0_image_full,
            path=self.vg_build_config.OUTPUT_FOLDER_PATH
            / f"images/6directions-LOD{self.lod_level if self.lod_level != None else 's'}.png",
        )

        # [Render oak asset for ]

        R_from_json = torch.tensor(
            [
                [
                    -0.2704063653945923,
                    -0.1467125415802002,
                    0.9515018463134766,
                ],
                [
                    0.962746262550354,
                    -0.041207194328308105,
                    0.26724815368652344,
                ],
                [
                    5.960464477539063e-08,
                    0.9883204698562622,
                    0.15238964557647705,
                ],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        R_colmap = R_from_json
        R_colmap[:, 1:3] *= -1.0

        image_current_layer_under_specific_camera_pose = (
            consolited_cluster_of_current_layer.render(
                camera=Camera(
                    image_width=1600,
                    image_height=1600,
                    focal_x=1600 / 2 / math.tan(0.785398 / 2),
                    focal_y=1600 / 2 / math.tan(0.785398 / 2),
                    R=R_colmap,
                    t=torch.tensor(
                        [0.76105135679245, 0.15553927421569824, 0.19130465388298035],
                        device="cuda",
                        dtype=torch.float32,
                    ),
                )
            )
        )

        UTILITY.SaveImage(
            image=image_current_layer_under_specific_camera_pose,
            path=self.vg_build_config.OUTPUT_FOLDER_PATH
            / f"images/LOD{self.lod_level if self.lod_level != None else 's'}_given-pose.png",
        )


class Cluster:
    def __init__(
        self,
        # [set when instantialized]
        count: int,
        positions: torch.Tensor,
        scales: torch.Tensor,
        quaternions: torch.Tensor,
        opacities: torch.Tensor,
        rgbs: torch.Tensor,
        # Assign an int if all 3D Gaussians in the cluster are at the same level.
        lod_level: int | None = None,
        # [set after instantialized]
        # for selection
        child_center_in_cluster_group: list[float] = [0.0, 0.0, 0.0],
        parent_center_in_cluster_group: list[float] = [math.inf, math.inf, math.inf],
        child_radius_in_cluster_group: float = -1.0,
        parent_radius_in_cluster_group: float = math.inf,
        # [config]
        vg_build_config: VGBuildConfig = None,
    ) -> None:
        """
        This function is the only entrance for creating Cluster instance.
        All other setup methods should call this function and return a Cluster.
        """

        self.vg_build_config = vg_build_config

        # count of 3D Gaussians
        self.count: int = count

        self.positions: torch.Tensor = positions
        self.scales: torch.Tensor = scales
        self.quaternions: torch.Tensor = quaternions
        self.opacities: torch.Tensor = opacities
        # only rgb (sh0) is considered in this project
        self.rgbs: torch.Tensor = rgbs

        self.lod_level: int | None = lod_level

        self.child_center_in_cluster_group: list[float] = child_center_in_cluster_group
        self.parent_center_in_cluster_group: list[float] = (
            parent_center_in_cluster_group
        )
        self.child_radius_in_cluster_group: float = child_radius_in_cluster_group
        self.parent_radius_in_cluster_group: float = parent_radius_in_cluster_group

    def splitIntoClusters(self) -> Clusters:
        if (
            self.count
            > self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER
        ):
            all_complete_clusters: list[Cluster] = []
            all_incomplete_clusters: list[Cluster] = [self]

            while len(all_incomplete_clusters) != 0:
                current_incomplete_clusters: list[Cluster] = []
                for incomplete_cluster in all_incomplete_clusters:
                    lengths = torch.tensor(
                        [
                            incomplete_cluster.positions[:, 0].max().item()
                            - incomplete_cluster.positions[:, 0].min().item(),
                            incomplete_cluster.positions[:, 1].max().item()
                            - incomplete_cluster.positions[:, 1].min().item(),
                            incomplete_cluster.positions[:, 2].max().item()
                            - incomplete_cluster.positions[:, 2].min().item(),
                        ]
                    )
                    axis_to_split = lengths.argmax().item()
                    axis_median = (
                        incomplete_cluster.positions[:, axis_to_split].median().item()
                    )

                    indices_left = (
                        incomplete_cluster.positions[:, axis_to_split] <= axis_median
                    )
                    indices_right = (
                        incomplete_cluster.positions[:, axis_to_split] > axis_median
                    )

                    cluster_left = Cluster(
                        vg_build_config=self.vg_build_config,
                        count=indices_left.sum().item(),
                        positions=incomplete_cluster.positions[indices_left],
                        scales=incomplete_cluster.scales[indices_left],
                        quaternions=incomplete_cluster.quaternions[indices_left],
                        opacities=incomplete_cluster.opacities[indices_left],
                        rgbs=incomplete_cluster.rgbs[indices_left],
                        lod_level=incomplete_cluster.lod_level,
                    )
                    cluster_right = Cluster(
                        vg_build_config=self.vg_build_config,
                        count=indices_right.sum().item(),
                        positions=incomplete_cluster.positions[indices_right],
                        scales=incomplete_cluster.scales[indices_right],
                        quaternions=incomplete_cluster.quaternions[indices_right],
                        opacities=incomplete_cluster.opacities[indices_right],
                        rgbs=incomplete_cluster.rgbs[indices_right],
                        lod_level=incomplete_cluster.lod_level,
                    )
                    if (
                        cluster_left.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER
                    ):
                        all_complete_clusters.append(cluster_left)
                    else:
                        current_incomplete_clusters.append(cluster_left)

                    if (
                        cluster_right.count
                        <= self.vg_build_config.BUILD_APPROPRIATE_COUNT_OF_GAUSSIANS_IN_ONE_CLUSTER
                    ):
                        all_complete_clusters.append(cluster_right)
                    else:
                        current_incomplete_clusters.append(cluster_right)

                    # ExLog(
                    #     f"{len(all_complete_clusters)=} {len(all_incomplete_clusters)=} {len(current_incomplete_clusters)=} {incomplete_cluster.count=} {cluster_left.count=} {cluster_right.count=}"
                    # )

                all_incomplete_clusters = current_incomplete_clusters

            return Clusters(
                vg_build_config=self.vg_build_config,
                clusters=all_complete_clusters,
                lod_level=self.lod_level,
            )
        else:
            clusters: list[Cluster] = [self]
            return Clusters(
                vg_build_config=self.vg_build_config,
                clusters=clusters,
                lod_level=self.lod_level,
            )

    def buildAllLodLayers(self) -> ClustersList:
        """
        input: primitives in gsply of the 3DGS asset
        output: all lod layers for this asset

        This function should be called only once on primitives read from gsply / LOD0.
        """

        ExLog(f"input gaussians -> all lod layers...")

        # [primitives -> LOD0 clusters]

        ExLog(f"input gaussians -> LOD0 clusters...")

        with ExTimer("splitIntoClusters()"):
            lod0: Clusters = self.splitIntoClusters()
            lod0.updatePositionsOfAllClusters()
        ExLog(
            f"split primitives into LOD0 clusters: {self.count} gaussians -> {lod0.count} clusters.",
        )

        lod0.savePlyWithDifferentColors(
            path=self.vg_build_config.OUTPUT_FOLDER_PATH
            / f"plys/lod{lod0.lod_level}-clusters.ply"
        )
        lod0.saveUsefulImages()

        # [LODx -> LODx+1]

        lod_layers: ClustersList = ClustersList(
            vg_build_config=self.vg_build_config, clusters_list=[lod0]
        )

        current_lod_layer: Clusters = lod0
        while (
            current_lod_layer.count
            > self.vg_build_config.BUILD_MAX_COUNT_OF_CLUSTERS_IN_COARSEST_LOD_LAYER
        ):
            print()
            ExLog(
                f"LOD{current_lod_layer.lod_level} -> LOD{current_lod_layer.lod_level + 1}...",
            )
            coarser_lod_layer = current_lod_layer.buildCoarserLodLayer()
            coarser_lod_layer.updatePositionsOfAllClusters()
            ExLog(
                f"Simplification: {current_lod_layer.count} clusters -> {coarser_lod_layer.count} clusters.",
            )

            # [save useful things]

            coarser_lod_layer.savePlyWithDifferentColors(
                path=self.vg_build_config.OUTPUT_FOLDER_PATH
                / f"plys/lod{coarser_lod_layer.lod_level}-clusters.ply"
            )
            coarser_lod_layer.saveUsefulImages()

            # [save useful things - ends]

            lod_layers.append(coarser_lod_layer)
            current_lod_layer = coarser_lod_layer

        return lod_layers

    def downSampleHalf(self) -> Cluster:
        if (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "random+s213"
        ):
            important_indices = torch.randperm(self.count)[: self.count // 2]
            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 3))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                lod_level=self.lod_level + 1,
            )
        elif (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "o+s213"
        ):
            # [only keeps 3D Gaussians with large opacities]
            integral_opacities = self.opacities

            descending_indices = integral_opacities.argsort(dim=0, descending=True)[
                :, 0
            ]
            important_indices = descending_indices[: self.count // 2]

            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 3))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                lod_level=self.lod_level + 1,
            )
        elif (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "osss23+s216"
        ):
            # [integral opacity: only keep gaussians with large scales and opacity]

            areas = torch.prod(self.scales, dim=1, keepdim=True) ** (2 / 3)
            integral_opacities = self.opacities * areas

            descending_indices = integral_opacities.argsort(dim=0, descending=True)[
                :, 0
            ]
            important_indices = descending_indices[: self.count // 2]

            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 6))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                lod_level=self.lod_level + 1,
            )
        elif (
            self.vg_build_config.SIMPLIFICATION_INITIALIZATION_DOWNSAMPLE_STRATEGY
            == "voxels+osss23+s216"
        ):
            median_x = self.positions[:, 0].median()
            median_y = self.positions[:, 1].median()
            median_z = self.positions[:, 2].median()

            xp, xn = self.positions[:, 0] >= median_x, self.positions[:, 0] < median_x
            yp, yn = self.positions[:, 1] >= median_y, self.positions[:, 1] < median_y
            zp, zn = self.positions[:, 2] >= median_z, self.positions[:, 2] < median_z

            eight_voxels_indices = [
                torch.where(xp & yp & zp)[0],
                torch.where(xp & yp & zn)[0],
                torch.where(xp & yn & zp)[0],
                torch.where(xp & yn & zn)[0],
                torch.where(xn & yp & zp)[0],
                torch.where(xn & yp & zn)[0],
                torch.where(xn & yn & zp)[0],
                torch.where(xn & yn & zn)[0],
            ]

            eight_important_indices = []

            for i in range(8):
                indices_to_select = eight_voxels_indices[i]

                # [only keeps 3D Gaussians with large opacities]
                areas = torch.prod(
                    self.scales[indices_to_select], dim=1, keepdim=True
                ) ** (2 / 3)
                integral_opacities = self.opacities[indices_to_select] * areas

                descending_indices = integral_opacities.argsort(dim=0, descending=True)[
                    :, 0
                ]
                important_indices = descending_indices[
                    : indices_to_select.numel() // 2 + 1
                ]

                eight_important_indices.append(indices_to_select[important_indices])

            important_indices = torch.cat(eight_important_indices)

            return Cluster(
                vg_build_config=self.vg_build_config,
                count=important_indices.numel(),
                positions=self.positions[important_indices],
                scales=(
                    self.scales[important_indices] * (2.0 ** (1 / 6))
                    if self.vg_build_config.SIMPLIFICATION_INITIALIZATION_SCALE_EXPANSION
                    else self.scales[important_indices]
                ),
                quaternions=self.quaternions[important_indices],
                opacities=self.opacities[important_indices],
                rgbs=self.rgbs[important_indices],
                lod_level=self.lod_level + 1,
            )
        else:
            ExLog("no strategy for downsampling", "ERROR")
            exit(-1)

    def optimizeUsingLocalSplatting(self, cluster_original: Cluster) -> Cluster:
        """
        For an original cluster group, with `self.count` 3D Gaussians, first down sample half and increase their scales to derive the initial simplified cluster group. Then use `LearnableGaussians` to optimize properties of newly created 3D Gaussians to distill the appearance of the original cluster group.
        """
        cluster_original_center = cluster_original.getCenter()
        cluster_original_cluster_radius = (
            cluster_original.getFarthestDistanceInClusterGroup()
        )
        cluster_original_cluster_mean_scale = (
            cluster_original.getMeanScaleInClusterGroup()
        )
        # ExLog(
        #     "DEBUG",
        #     f"{cluster_original_center=} {cluster_original_cluster_radius=:.4f} {cluster_original_cluster_mean_scale=:.4f}",
        # )

        # [cluster_simplified_init]

        cluster_simplified_init = self

        # [optimize to simplify cluster_original and derive cluster_simplified]

        learnable_cluster = LearnableGaussians(
            vg_build_config=self.vg_build_config,
            original_gaussians=cluster_original,
            gaussians=cluster_simplified_init,
            center=cluster_original_center,
            distance=cluster_original_cluster_radius * 4.0,
        )
        learnable_cluster.train()
        cluster_simplified = learnable_cluster.toGaussians(lod_level=self.lod_level)

        return cluster_simplified

    def render(self, camera: Camera) -> torch.Tensor:
        # (b, h, w, 3), (b, h, w, 1)
        image_rgb_premultiplied, gsplat_image_a, statistics = gsplat.rasterization(
            means=self.positions,
            quats=self.quaternions,
            scales=self.scales,
            opacities=self.opacities[:, 0],
            sh_degree=None,
            colors=self.rgbs,
            viewmats=camera.view_matrix.T[
                None, :, :
            ],  # Camera class uses row vectors, while gsplat uses colume vectors.
            Ks=torch.tensor(
                [
                    [camera.focal_x, 0.0, camera.image_width / 2],
                    [0.0, camera.focal_y, camera.image_height / 2],
                    [0.0, 0.0, 1.0],
                ]
            )[None, :, :],
            width=camera.image_width,
            height=camera.image_height,
            eps2d=0.0,
        )

        # only render one image
        # (h, w, 3), (h, w, 1)
        image_rgb_premultiplied = image_rgb_premultiplied[0, ...]
        gsplat_image_a = gsplat_image_a[0, ...]

        # premultiplies rgb -> not premultiplied rgb
        mask_alpha_none_zero = (gsplat_image_a != 0.0)[..., 0]
        image_rgb_premultiplied[mask_alpha_none_zero] = image_rgb_premultiplied[
            mask_alpha_none_zero
        ] / gsplat_image_a[mask_alpha_none_zero].clip(max=1.0)

        # move color channel to the first position
        # (3, h, w), (1, h, w,)
        image_rgb_premultiplied = einops.rearrange(
            image_rgb_premultiplied, "h w c -> c h w"
        )
        gsplat_image_a = einops.rearrange(gsplat_image_a, "h w c -> c h w")

        # concatenate together
        # (4, h, w)
        image_rgba = torch.concat([image_rgb_premultiplied, gsplat_image_a], dim=0)

        return image_rgba

    def renderReturnCountAndDuration(
        self,
        camera: Camera,
        gsplat_radius_clip: float = 0.0,
    ) -> tuple[torch.Tensor, int, float]:
        """
        return:
            torch.Tensor: image (4 w h)
            int: number of gaussians in the frustum
            float: duration to rasterize
        """

        # NOTICE only use for vg-select fps metrics, should remove when do vg-build
        for i in range(5):
            if i == 4:
                time_start_gsplat: float = time.perf_counter()
            # (b, h, w, 3), (b, h, w, 1)
            image_rba_premultiplied, gsplat_image_a, statistics = gsplat.rasterization(
                means=self.positions,
                quats=self.quaternions,
                scales=self.scales,
                opacities=self.opacities[:, 0],
                sh_degree=None,
                colors=self.rgbs,
                viewmats=camera.view_matrix.T[
                    None, :, :
                ],  # Camera class uses row vectors, while gsplat uses colume vectors.
                Ks=torch.tensor(
                    [
                        [camera.focal_x, 0.0, camera.image_width / 2],
                        [0.0, camera.focal_y, camera.image_height / 2],
                        [0.0, 0.0, 1.0],
                    ]
                )[None, :, :],
                width=camera.image_width,
                height=camera.image_height,
                eps2d=0.0,
                radius_clip=gsplat_radius_clip,
            )
            if i == 4:
                time_end_gsplat: float = time.perf_counter()
                time_duration_gsplat: float = time_end_gsplat - time_start_gsplat

        number_of_gaussians_in_frustum: int = statistics["gaussian_ids"].shape[0]
        number_of_duplicated_gaussians: int = statistics["flatten_ids"].shape[0]

        # only render one image
        # (h, w, 3), (h, w, 1)
        image_rba_premultiplied = image_rba_premultiplied[0, ...]
        gsplat_image_a = gsplat_image_a[0, ...]

        # premultiplies rgb -> not premultiplied rgb
        mask_alpha_none_zero = (gsplat_image_a != 0.0)[..., 0]
        image_rba_premultiplied[mask_alpha_none_zero] = image_rba_premultiplied[
            mask_alpha_none_zero
        ] / gsplat_image_a[mask_alpha_none_zero].clip(max=1.0)

        # move color channel to the first position
        # (3, h, w), (1, h, w,)
        image_rba_premultiplied = einops.rearrange(
            image_rba_premultiplied, "h w c -> c h w"
        )
        gsplat_image_a = einops.rearrange(gsplat_image_a, "h w c -> c h w")

        # concatenate together
        # (4, h, w)
        image_rgba = torch.concat([image_rba_premultiplied, gsplat_image_a], dim=0)

        return (
            image_rgba,
            number_of_gaussians_in_frustum,
            number_of_duplicated_gaussians,
            time_duration_gsplat,
        )

    def renderFullImageConsolidatingSixDirections(
        self, center: torch.Tensor, distance: float
    ) -> torch.Tensor:
        cameras: list[Camera] = Camera.DeriveSixDirections(
            center=center, distance=distance
        )

        image_rgb_full = torch.zeros(
            (4, cameras[0].image_height * 3, cameras[0].image_width * 4),
            dtype=torch.float,
        )
        for i, camera in enumerate(cameras):
            image_rgb = self.render(camera=camera)
            if i == 0:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    1 * camera.image_width : 2 * camera.image_width,
                ] = image_rgb
            elif i == 1:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    0 * camera.image_width : 1 * camera.image_width,
                ] = image_rgb
            elif i == 2:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    2 * camera.image_width : 3 * camera.image_width,
                ] = image_rgb
            elif i == 3:
                image_rgb_full[
                    :,
                    1 * camera.image_height : 2 * camera.image_height,
                    3 * camera.image_width : 4 * camera.image_width,
                ] = image_rgb
            elif i == 4:
                image_rgb_full[
                    :,
                    0 * camera.image_height : 1 * camera.image_height,
                    1 * camera.image_width : 2 * camera.image_width,
                ] = image_rgb
            elif i == 5:
                image_rgb_full[
                    :,
                    2 * camera.image_height : 3 * camera.image_height,
                    1 * camera.image_width : 2 * camera.image_width,
                ] = image_rgb
        return image_rgb_full

    def getCenter(self) -> torch.Tensor:
        """
        return (1,3)
        """
        return self.positions.mean(dim=0, keepdim=True)

    def getFarthestDistanceInClusterGroup(self) -> float:
        distance: float = (
            torch.sqrt(
                torch.pow(
                    (self.positions - self.getCenter()),
                    2,
                ).sum(dim=1)
            )
            .max()
            .item()
        )
        return distance

    def getMeanScaleInClusterGroup(self) -> float:
        return self.scales.mean(dim=0).mean().item()

    def getRadiusValueInClusterGroupForSelection(self) -> float:
        """
        Use a bounding sphere to put all gaussians in self(cluster) inside. Return the value of the radius of the bounding sphere.

        Should use after simplification in a cluster group.
        """
        distance: float = self.getFarthestDistanceInClusterGroup()
        # TODO not aligns with paper
        mean_scale: float = self.getMeanScaleInClusterGroup()
        # 240806-1337 remove scale
        radius = distance  # + 3.0 * mean_scale  # 99%
        return radius

    def saveActivatedPly(self, path: pathlib.Path) -> None:
        """
        Different from vanilla 3DGS, we activate all the learnable parameters and save the physical properties of 3D Gaussians in ply files.
        """

        # TODO add lod level, frequency
        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                self.scales.cpu().numpy(),
                self.quaternions.cpu().numpy(),
                self.opacities.cpu().numpy(),
                # Save using the original precision. To visualize the color, convert this value to uint8/u1.
                self.rgbs.cpu().numpy(),
            ],
            axis=1,
        )
        ply_properties = (
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
            ]
            + [
                ("scales_0", "f4"),
                ("scales_1", "f4"),
                ("scales_2", "f4"),
            ]
            + [
                ("quaternions_0", "f4"),
                ("quaternions_1", "f4"),
                ("quaternions_2", "f4"),
                ("quaternions_3", "f4"),
            ]
            + [("opacities", "f4")]
            + [
                ("rgbs_0", "f4"),
                ("rgbs_1", "f4"),
                ("rgbs_2", "f4"),
            ]
        )

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )
        ExLog(f"Saved {self.count} points to {path}.")

    def saveOriginalPly(self, path: pathlib.Path) -> None:
        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                LearnableGaussians.InverseActivationScales(self.scales).cpu().numpy(),
                self.quaternions.cpu().numpy(),
                LearnableGaussians.InverseActivationOpacities(self.opacities)
                .cpu()
                .numpy(),
                # Save using the original precision. To visualize the color, convert this value to uint8/u1.
                LearnableGaussians.InverseActivationSh0(self.rgbs).cpu().numpy(),
            ],
            axis=1,
        )
        ply_properties = (
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
            ]
            + [
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
            ]
            + [
                ("rot_0", "f4"),
                ("rot_1", "f4"),
                ("rot_2", "f4"),
                ("rot_3", "f4"),
            ]
            + [("opacity", "f4")]
            + [
                ("f_dc_0", "f4"),
                ("f_dc_1", "f4"),
                ("f_dc_2", "f4"),
            ]
        )

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )
        ExLog(f"Saved {self.count} points to {path}.")

    def savePlyWithClusterColors(
        self,
        path: pathlib.Path,
        k: int,
        labels: np.ndarray,
    ) -> None:
        color_choices = np.random.randint(low=0, high=255, size=(k, 3), dtype=np.uint8)
        colors = np.zeros((self.count, 3), dtype=np.int8)
        colors = color_choices[labels]

        ply_points = np.concatenate(
            [
                self.positions.cpu().numpy(),
                colors,
            ],
            axis=1,
        )
        ply_properties = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ] + [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        UTILITY.SavePlyUsingPlyfilePackage(
            path=path,
            points=ply_points,
            properties=ply_properties,
        )

    # [select]

    def transformed(
        self,
        scale_factor: float,
        R: torch.Tensor,
        t: torch.Tensor,
    ) -> Cluster:
        Rt_matrices_original = torch.zeros((self.count, 4, 4), dtype=torch.float32)
        Rt_matrices_original[:, 3, 3] = 1.0
        Rt_matrices_original[:, :3, 3] = self.positions
        # NOTICE: r xyz -> xyz w
        Rt_matrices_original[:, :3, :3] = torch.from_numpy(
            scipy.spatial.transform.Rotation.from_quat(
                self.quaternions.cpu().numpy(),
                scalar_first=True,
            ).as_matrix()
        )

        # random scale (notice that we only use uniform scaling on xyz the same time)
        S1 = torch.eye(4, dtype=torch.float32)
        S1[:3] *= scale_factor
        new_scales = self.scales * scale_factor

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

        return Cluster(
            vg_build_config=self.vg_build_config,
            count=self.count,
            positions=new_positions,
            scales=new_scales,
            quaternions=new_quaternions,
            opacities=self.opacities.clone(),
            rgbs=self.rgbs.clone(),
        )

    def selectedInFrustum(self, camera: Camera) -> Cluster:
        # (Gaussians_Count, 4)
        Gaussians_Positions_Homogeneous = torch.zeros(
            (self.count, 4),
            dtype=torch.float32,
        )
        Gaussians_Positions_Homogeneous[:, :3] = self.positions
        Gaussians_Positions_Homogeneous[:, 3] = 1.0

        # (Gaussians_Count, 4)
        Gaussians_Positions_Viewed = (
            Gaussians_Positions_Homogeneous @ camera.view_matrix
        )

        # (Gaussians_Count, 4)
        Gaussians_Positions_Projected = (
            Gaussians_Positions_Viewed @ camera.projection_matrix
        )
        Gaussians_Positions_Projected = Gaussians_Positions_Projected * (
            1.0 / (Gaussians_Positions_Projected[:, -1:] + 0.0000001)
        )

        selected_mask = (
            (Gaussians_Positions_Projected[:, 0] > -1.1)
            & (Gaussians_Positions_Projected[:, 0] < 1.1)
            & (Gaussians_Positions_Projected[:, 1] > -1.1)
            & (Gaussians_Positions_Projected[:, 1] < 1.1)
            & (Gaussians_Positions_Viewed[:, 2] > 0.0)
        )
        ExLog(
            f"{Gaussians_Positions_Projected.shape=} {selected_mask.shape=} {selected_mask.sum().item()=}"
        )

        return Cluster(
            count=selected_mask.sum().item(),
            positions=self.positions[selected_mask, :],
            scales=self.scales[selected_mask, :],
            quaternions=self.quaternions[selected_mask, :],
            opacities=self.opacities[selected_mask, :],
            rgbs=self.rgbs[selected_mask, :],
        )


class GsPly:
    def Normalize(x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).sum(dim=1).sqrt()[:, None]
        return x / norm

    def Sigmoid(x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + torch.exp(-x))

    def Sh0ToRgb(sh0: torch.Tensor) -> torch.Tensor:
        SH_C0 = 0.28209479177387814
        return torch.clip(sh0 * SH_C0 + 0.5, min=0.0, max=None)

    def __init__(
        self,
        vg_build_config: VGBuildConfig,
    ) -> None:
        self.vg_build_config = vg_build_config

    def read(self) -> Cluster:
        """
        Read activated values / physical properties from gsply.
        """
        points: plyfile.PlyElement = plyfile.PlyData.read(
            self.vg_build_config.ASSET_GSPLY_PATH
        )["vertex"]
        ExLog(
            f"Read {points.count} points from {self.vg_build_config.ASSET_GSPLY_PATH}."
        )

        gsply_positions: np.ndarray = np.column_stack(
            (
                points["x"],
                points["y"],
                points["z"],
            )
        )
        gsply_scales: np.ndarray = np.column_stack(
            (
                points["scale_0"],
                points["scale_1"],
                points["scale_2"],
            )
        )
        gsply_quaternions: np.ndarray = np.column_stack(
            (
                points["rot_0"],
                points["rot_1"],
                points["rot_2"],
                points["rot_3"],
            )
        )
        gsply_opacities: np.ndarray = np.column_stack((points["opacity"],)).astype(
            np.float32
        )
        gsply_sh0s: np.ndarray = np.column_stack(
            (points["f_dc_0"], points["f_dc_1"], points["f_dc_2"])
        )

        return Cluster(
            vg_build_config=self.vg_build_config,
            count=points.count,
            positions=torch.tensor(gsply_positions, dtype=torch.float32),
            scales=torch.exp(torch.tensor(gsply_scales, dtype=torch.float32)),
            quaternions=GsPly.Normalize(
                torch.tensor(gsply_quaternions, dtype=torch.float32)
            ),
            opacities=GsPly.Sigmoid(torch.tensor(gsply_opacities, dtype=torch.float32)),
            rgbs=GsPly.Sh0ToRgb(torch.tensor(gsply_sh0s, dtype=torch.float32)),
            lod_level=0,
        )
