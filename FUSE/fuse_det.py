def make_Ramp(ramp_colors):
    from colour import Color
    from matplotlib.colors import LinearSegmentedColormap

    color_ramp = LinearSegmentedColormap.from_list(
        "my_list", [Color(c1).rgb for c1 in ramp_colors]
    )
    return color_ramp


import numpy as np

custom_ramp = make_Ramp(["#000000", "#D62728"])
red = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#FF7F0E"])
orange = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#17BECF"])
blue = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#2CA02C"])
green = custom_ramp(range(256))[:, :-1] * 255

custom_ramp = make_Ramp(["#000000", "#9467BD"])
purple = custom_ramp(range(256))[:, :-1] * 255

red = red.astype(np.uint8).T
orange = orange.astype(np.uint8).T
blue = blue.astype(np.uint8).T
purple = purple.astype(np.uint8).T
green = green.astype(np.uint8).T

from FUSE.fuse_illu import FUSE_illu
from FUSE.utils import (
    # GuidedFilter, # unused import
    sgolay2dkernel,
    waterShed,
    refineShape,
    # extendBoundary, # unused import
    EM2DPlus,
    fusion_perslice,
    extendBoundary2,
    imagej_metadata_tags,
)
from skimage import measure
from FUSE.NSCT import NSCTdec #, NSCTrec # unused import
from typing import Union, Dict #, Tuple, Optional, List # unused import
import dask
import torch
import os
from aicsimageio import AICSImage
# import scipy.ndimage as ndimage # unused import
import ants
import scipy.io as scipyio
import cupy
from cucim.skimage import feature
from cupyx.scipy.ndimage import map_coordinates as map_coordinates
# import cucim # unused import
import open3d as o3d
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import morphology
import shutil
# import io # unused import
import copy
import SimpleITK as sitk
import tqdm
import gc
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.width", 10000)
import matplotlib.patches as patches
import torch.nn as nn
import torch.nn.functional as F
import tifffile


class FUSE_det:
    def __init__(
        self,
        require_precropping: bool = True,
        precropping_params: list[int, int, int, int] = [],
        resample_ratio: int = 2,
        window_size: list[int, int] = [5, 59],
        poly_order: list[int, int] = [2, 2],
        n_epochs: int = 50,
        require_segmentation: bool = True,
        skip_illuFusion: bool = True,
        destripe_preceded: bool = False,
        destripe_params: Dict = None,
        device: str = "cuda",
    ):
        if !(device == "cuda" and torch.cuda.is_available()):
            device = "cpu"
        self.train_params = {
            "require_precropping": require_precropping,
            "precropping_params": precropping_params,
            "resample_ratio": resample_ratio,
            "window_size": window_size,
            "poly_order": poly_order,
            "n_epochs": n_epochs,
            "require_segmentation": require_segmentation,
            "device": device,
        }
        self.modelFront = FUSE_illu(**self.train_params)
        self.modelBack = FUSE_illu(**self.train_params)
        self.train_params.update(
            {
                "skip_illuFusion": skip_illuFusion,
                "destripe_preceded": destripe_preceded,
                "destripe_params": destripe_params,
            }
        )
        self.train_params["kernel2d"] = (
            torch.from_numpy(
                sgolay2dkernel(
                    np.array([window_size[1], window_size[1]]),
                    np.array(
                        [
                            self.train_params["poly_order"][1],
                            self.train_params["poly_order"][1],
                        ]
                    ),
                )
            )
            .to(torch.float)
            .to(self.train_params["device"])
        )

    def train(
        self,
        require_registration: bool,
        require_flipping_along_illu_for_dorsaldet: bool,
        require_flipping_along_det_for_dorsaldet: bool,
        data_path: str = "",
        sample_name: str = "",
        sparse_sample=False,
        top_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        top_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        bottom_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_ventral_det_data: Union[dask.array.core.Array, str] = None,
        left_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        right_illu_dorsal_det_data: Union[dask.array.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
        save_separate_results: bool = False,
        z_spacing: float = None,
        xy_spacing: float = None,
    ):
        if require_registration:
            if (z_spacing == None) or (xy_spacing == None):
                print("spacing information is missing.")
                return
        illu_name = "illuFusionResult{}".format(
            "" if self.train_params["require_segmentation"] else "_without_segmentation"
        )
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return
        save_path = os.path.join(save_path, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        f_flag = 0
        flip_axes = []
        if require_flipping_along_det_for_dorsaldet:
            flip_axes.append(0)
            f_flag = 1
        if require_flipping_along_illu_for_dorsaldet:
            flip_axes.append(1)
            f_flag = 1
        flip_axes = tuple(flip_axes)
        self.sample_params = {
            "require_registration": require_registration,
            "require_flipping_along_det_for_dorsaldet": require_flipping_along_det_for_dorsaldet,
            "require_flipping_along_illu_for_dorsaldet": require_flipping_along_illu_for_dorsaldet,
            "z_spacing": z_spacing,
            "xy_spacing": xy_spacing,
        }

        if (
            (top_illu_ventral_det_data is not None)
            and (bottom_illu_ventral_det_data is not None)
            and (top_illu_dorsal_det_data is not None)
            and (bottom_illu_dorsal_det_data is not None)
        ):
            T_flag = 0
            if isinstance(top_illu_ventral_det_data, str):
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(top_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    "top_illu+ventral_det"
                )
            if isinstance(bottom_illu_ventral_det_data, str):
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(bottom_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    "bottom_illu+ventral_det"
                )
            if isinstance(top_illu_dorsal_det_data, str):
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(top_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    "top_illu+dorsal_det"
                )
            if isinstance(bottom_illu_dorsal_det_data, str):
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(bottom_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    "bottom_illu+dorsal_det"
                )
        elif (
            (left_illu_ventral_det_data is not None)
            and (right_illu_ventral_det_data is not None)
            and (left_illu_dorsal_det_data is not None)
            and (right_illu_dorsal_det_data is not None)
        ):
            T_flag = 1
            if isinstance(left_illu_ventral_det_data, str):
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(left_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    "left_illu+ventral_det"
                )
            if isinstance(right_illu_ventral_det_data, str):
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(right_illu_ventral_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_ventraldet_data_saving_name"] = (
                    "right_illu+ventral_det"
                )
            if isinstance(left_illu_dorsal_det_data, str):
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(left_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    "left_illu+dorsal_det"
                )
            if isinstance(right_illu_dorsal_det_data, str):
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(right_illu_dorsal_det_data)[0]
                )
            else:
                self.sample_params["bottomillu_dorsaldet_data_saving_name"] = (
                    "right_illu+dorsal_det"
                )
        else:
            print("input(s) missing, please check.")
            return

        for k in self.sample_params.keys():
            if "saving_name" in k:
                sub_folder = os.path.join(save_path, self.sample_params[k])
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)

        if self.train_params["skip_illuFusion"]:
            if os.path.exists(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    illu_name + ".tif",
                )
            ):
                print("Skip dual-illu fusion for ventral det...")
                illu_flag_ventral = 0
            else:
                print("Cannot skip dual-illu fusion for ventral det...")
                illu_flag_ventral = 1
            if os.path.exists(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    illu_name + ".tif",
                )
            ):
                print("Skip dual-illu fusion for dorsal det...")
                illu_flag_dorsal = 0
            else:
                print("Cannot skip dual-illu fusion for dorsal det...")
                illu_flag_dorsal = 1
        else:
            illu_flag_dorsal = 1
            illu_flag_ventral = 1

        if illu_flag_ventral:
            print("\nFusion along illumination for ventral camera...")
            self.modelFront.train(
                data_path=data_path,
                sample_name=sample_name,
                top_illu_data=top_illu_ventral_det_data,
                bottom_illu_data=bottom_illu_ventral_det_data,
                left_illu_data=left_illu_ventral_det_data,
                right_illu_data=right_illu_ventral_det_data,
                save_path=save_path,
                save_folder="",
                save_separate_results=save_separate_results,
                sparse_sample=sparse_sample,
                camera_position="ventral_det",
            )
        if illu_flag_dorsal:
            print("\nFusion along illumination for dorsal camera...")
            self.modelBack.train(
                data_path=data_path,
                sample_name=sample_name,
                top_illu_data=top_illu_dorsal_det_data,
                bottom_illu_data=bottom_illu_dorsal_det_data,
                left_illu_data=left_illu_dorsal_det_data,
                right_illu_data=right_illu_dorsal_det_data,
                save_path=save_path,
                save_folder="",
                save_separate_results=save_separate_results,
                sparse_sample=sparse_sample,
                cam_pos=(
                    "back"
                    if require_flipping_along_det_for_dorsaldet == False
                    else "front"
                ),
                camera_position="dorsal_det",
            )

        data_path = os.path.join(data_path, sample_name)

        if require_registration:
            if not os.path.exists(
                os.path.join(
                    save_path, self.sample_params["topillu_ventraldet_data_saving_name"]
                )
                + "/regInfo{}.npy".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe"
                )
            ):
                print("\nRegister...")
                print("read in...")
                if self.train_params["destripe_preceded"]:
                    illu_name_ = illu_name + "+RESULT/{}.tif".format(
                        self.train_params["destripe_params"]
                    )
                else:
                    illu_name_ = illu_name + ".tif"
                respective_view_uint16_handle = AICSImage(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        illu_name_,
                    )
                )
                moving_view_uint16_handle = AICSImage(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_dorsaldet_data_saving_name"],
                        illu_name_,
                    )
                )
                respective_view_uint16 = respective_view_uint16_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
                moving_view_uint16 = moving_view_uint16_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )

                if f_flag:
                    moving_view_uint16[:] = np.flip(moving_view_uint16, flip_axes)
                s_r, m, n = respective_view_uint16.shape
                s_m, _, _ = moving_view_uint16.shape
                if s_r == s_m:
                    moving_view_uint16_pad = copy.deepcopy(moving_view_uint16)
                    respective_view_uint16_pad = copy.deepcopy(respective_view_uint16)
                elif s_r > s_m:
                    moving_view_uint16_pad = np.concatenate(
                        (
                            np.zeros((s_r - s_m, m, n), dtype=moving_view_uint16.dtype),
                            moving_view_uint16,
                        ),
                        0,
                    )
                    respective_view_uint16_pad = copy.deepcopy(respective_view_uint16)
                else:
                    respective_view_uint16_pad = np.concatenate(
                        (
                            respective_view_uint16,
                            np.zeros((s_m - s_r, m, n), dtype=moving_view_uint16.dtype),
                        ),
                        0,
                    )
                    moving_view_uint16_pad = copy.deepcopy(moving_view_uint16)
                del moving_view_uint16, respective_view_uint16

                print("reg in zx...")
                yMP_respective = respective_view_uint16_pad.max(2)
                yMP_moving = moving_view_uint16_pad.max(2)
                AffineMapZX = coarseRegistrationZX(
                    yMP_respective,
                    yMP_moving,
                )
                del yMP_respective, yMP_moving
                print("reg in y...")
                zMP_respective = respective_view_uint16_pad.max(0)
                zMP_moving = moving_view_uint16_pad.max(0)
                AffineMapZXY, frontMIP, backMIP = coarseRegistrationY(
                    zMP_respective,
                    zMP_moving,
                    AffineMapZX,
                )
                del zMP_respective, zMP_moving
                print("rigid registration in 3D...")
                a0, b0, c0, d0 = self.segMIP(frontMIP)
                a1, b1, c1, d1 = self.segMIP(backMIP)
                infomax = float(max(frontMIP.max(), backMIP.max()))
                xs, xe, ys, ye = min(c0, c1), max(d0, d1), min(a0, a1), max(b0, b1)
                x = min(
                    len(np.arange(0, xs)),
                    len(np.arange(xe, m)),
                )
                y = min(
                    len(np.arange(0, ys)),
                    len(np.arange(ye, n)),
                )
                xs, xe = x, -x if x != 0 else None
                ys, ye = y, -y if y != 0 else None

                reg_info = fineReg(
                    respective_view_uint16_pad,
                    moving_view_uint16_pad,
                    xs,
                    xe,
                    ys,
                    ye,
                    AffineMapZXY,
                    infomax,
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                    ),
                    self.train_params["destripe_preceded"],
                    z_spacing=self.sample_params["z_spacing"],
                    xy_spacing=self.sample_params["xy_spacing"],
                )
                del respective_view_uint16_pad, moving_view_uint16_pad
                reg_info.update(
                    {"zfront": s_r, "zback": s_m, "m": m, "n": n, "z": max(s_r, s_m)}
                )
                np.save(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "regInfo{}.npy".format(
                            ""
                            if (not self.train_params["destripe_preceded"])
                            else "_destripe"
                        ),
                    ),
                    reg_info,
                )

                if self.train_params["destripe_preceded"]:
                    illu_name_ = illu_name + "+RESULT/{}.tif".format(
                        self.train_params["destripe_params"]
                    )
                else:
                    illu_name_ = illu_name + ".tif"
                moving_view_uint16_handle = AICSImage(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_dorsaldet_data_saving_name"],
                        illu_name_,
                    )
                )
                moving_view_uint16 = moving_view_uint16_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
                regInfo = np.load(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "regInfo{}.npy".format(
                            ""
                            if (not self.train_params["destripe_preceded"])
                            else "_destripe"
                        ),
                    ),
                    allow_pickle=True,
                ).item()
                AffineMapZXY = regInfo["AffineMapZXY"]
                zfront = regInfo["zfront"]
                zback = regInfo["zback"]
                z = regInfo["z"]
                m = regInfo["m"]
                n = regInfo["n"]
                xcs, xce, ycs, yce = regInfo["region_for_reg"]
                padding_z = (
                    boundaryInclude(
                        regInfo,
                        (
                            z + int(np.ceil(-AffineMapZXY[0]))
                            if AffineMapZXY[0] < 0
                            else z
                        )
                        * self.sample_params["z_spacing"],
                        m
                        / 2
                        * self.train_params["resample_ratio"]
                        * self.sample_params["xy_spacing"],
                        n
                        / 2
                        * self.train_params["resample_ratio"]
                        * self.sample_params["xy_spacing"],
                        spacing=self.sample_params["z_spacing"],
                    )
                    / self.sample_params["z_spacing"]
                )
                trans_path = os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    illu_name
                    + "{}_coarse_reg.tif".format(
                        ""
                        if (not self.train_params["destripe_preceded"])
                        else "_destripe"
                    ),
                )
                volumeTranslate(
                    moving_view_uint16,
                    regInfo,
                    AffineMapZXY,
                    zback,
                    z,
                    m,
                    n,
                    padding_z,
                    trans_path,
                    T_flag,
                    f_flag,
                    flip_axes if f_flag else None,
                    device=self.train_params["device"],
                    resample_ratio=self.train_params["resample_ratio"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    z_spacing=self.sample_params["z_spacing"],
                )
            else:
                print("\nSkip registration...")

            if not os.path.exists(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    "reginfo_refine.npy",
                )
            ):
                print("refine registration...")
                if self.train_params["destripe_preceded"]:
                    illu_name_ = illu_name + "+RESULT/{}".format(
                        self.train_params["destripe_params"]
                    )
                else:
                    illu_name_ = illu_name
                respective_view_uint16_handle = AICSImage(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        illu_name_ + ".tif",
                    )
                )
                moving_view_uint16_handle = AICSImage(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_dorsaldet_data_saving_name"],
                        illu_name_ + "_coarse_reg.tif",
                    )
                )
                respective_view_uint16 = respective_view_uint16_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
                moving_view_uint16 = moving_view_uint16_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
                target_points = []
                source_points = []
                threshold_respective = filters.threshold_otsu(
                    respective_view_uint16[:, ::4, ::4]
                )
                for ind in tqdm.tqdm(
                    range(respective_view_uint16.shape[0]), leave=False, desc="DoG: "
                ):
                    tmp = feature.blob_dog(
                        cupy.asarray(respective_view_uint16[ind]),
                        min_sigma=1.8,
                        max_sigma=1.8 * 1.6 + 1,
                        threshold=0.001,
                    )[:, :-1]
                    if tmp.shape[0] != 0:
                        tmp = cupy.asnumpy(tmp)
                        tmp = np.concatenate(
                            (ind * np.ones((tmp.shape[0], 1)), tmp), 1
                        ).astype(np.int32)
                        mask = respective_view_uint16[ind] > threshold_respective
                        target_points.append(tmp[mask[tmp[:, 1], tmp[:, 2]], :])
                for ind in tqdm.tqdm(
                    range(moving_view_uint16.shape[0]), leave=False, desc="DoG: "
                ):
                    tmp = feature.blob_dog(
                        cupy.asarray(moving_view_uint16[ind]),
                        min_sigma=1.8,
                        max_sigma=1.8 * 1.6 + 1,
                        threshold=0.001,
                    )[:, :-1]
                    if tmp.shape[0] != 0:
                        tmp = cupy.asnumpy(tmp)
                        tmp = np.concatenate(
                            (ind * np.ones((tmp.shape[0], 1)), tmp), 1
                        ).astype(np.int32)
                        mask = moving_view_uint16[ind] > threshold_respective
                        source_points.append(tmp[mask[tmp[:, 1], tmp[:, 2]], :])
                source_points = np.concatenate(source_points, 0)
                target_points = np.concatenate(target_points, 0)
                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(
                    source_points
                    * np.array(
                        [
                            self.sample_params["z_spacing"]
                            / self.sample_params["xy_spacing"],
                            1,
                            1,
                        ]
                    )
                )
                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(
                    target_points
                    * np.array(
                        [
                            self.sample_params["z_spacing"]
                            / self.sample_params["xy_spacing"],
                            1,
                            1,
                        ]
                    )
                )
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    target_pcd,
                    source_pcd,
                    3.0,
                    np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(
                        with_scaling=True
                    ),
                    o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=5000
                    ),
                )
                np.save(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "reginfo_refine.npy",
                    ),
                    {
                        "source_points": source_points,
                        "target_points": target_points,
                        "transformation": reg_p2p.transformation,
                    },
                )
                volumeTranslate2(
                    moving_view_uint16,
                    reg_p2p.transformation,
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_dorsaldet_data_saving_name"],
                        illu_name_ + "_reg.tif",
                    ),
                    T_flag,
                    device=self.train_params["device"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    z_spacing=self.sample_params["z_spacing"],
                )
                regInfo = np.load(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "regInfo{}.npy".format(
                            ""
                            if (not self.train_params["destripe_preceded"])
                            else "_destripe"
                        ),
                    ),
                    allow_pickle=True,
                ).item()
                AffineMapZXY = regInfo["AffineMapZXY"]
                zback = regInfo["zback"]
                z = regInfo["z"]
                m = regInfo["m"]
                n = regInfo["n"]
                padding_z = (
                    boundaryInclude(
                        regInfo,
                        (
                            z + int(np.ceil(-AffineMapZXY[0]))
                            if AffineMapZXY[0] < 0
                            else z
                        )
                        * self.sample_params["z_spacing"],
                        m
                        / 2
                        * self.train_params["resample_ratio"]
                        * self.sample_params["xy_spacing"],
                        n
                        / 2
                        * self.train_params["resample_ratio"]
                        * self.sample_params["xy_spacing"],
                        spacing=self.sample_params["z_spacing"],
                    )
                    / self.sample_params["z_spacing"]
                )
                T2 = reg_p2p.transformation

                for f, f_name in zip(
                    ["top", "bottom"] if (not T_flag) else ["left", "right"],
                    ["top", "bottom"],
                ):
                    if not self.train_params["destripe_preceded"]:
                        if isinstance(locals()[f + "_illu_dorsal_det_data"], str):
                            f_handle = AICSImage(
                                os.path.join(
                                    data_path, locals()[f + "_illu_dorsal_det_data"]
                                )
                            )
                        else:
                            f_handle = AICSImage(locals()[f + "_illu_dorsal_det_data"])
                    else:
                        f0 = self.sample_params[
                            f_name + "illu_dorsaldet_data_saving_name"
                        ]
                        f_handle = AICSImage(
                            os.path.join(
                                save_path,
                                f0,
                                f0 + "+RESULT",
                                self.train_params["destripe_params"] + ".tif",
                            )
                        )
                    inputs = f_handle.get_image_data(
                        "ZXY" if T_flag else "ZYX", T=0, C=0
                    )
                    trans_path = os.path.join(
                        save_path,
                        self.sample_params[f_name + "illu_dorsaldet_data_saving_name"],
                        self.sample_params[f_name + "illu_dorsaldet_data_saving_name"]
                        + "{}_reg.tif".format(
                            ""
                            if (not self.train_params["destripe_preceded"])
                            else "_destripe"
                        ),
                    )
                    s_back = inputs.shape[0]
                    volumeTranslate_compose(
                        inputs,
                        regInfo,
                        AffineMapZXY,
                        T2,
                        zback,
                        z,
                        m,
                        n,
                        padding_z,
                        trans_path,
                        T_flag,
                        f_flag,
                        flip_axes if f_flag else None,
                        device=self.train_params["device"],
                        resample_ratio=self.train_params["resample_ratio"],
                        xy_spacing=self.sample_params["xy_spacing"],
                        z_spacing=self.sample_params["z_spacing"],
                    )

                fl = "fusionBoundary_xy{}".format(
                    ""
                    if self.train_params["require_segmentation"]
                    else "_without_segmentation"
                )
                f0 = os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    fl + ".tif",
                )
                boundary = tifffile.imread(f0).astype(np.float32)
                mask = np.arange(m)[None, :, None] > boundary[:, None, :]
                trans_path = os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    fl
                    + "{}_reg.npy".format(
                        ""
                        if (not self.train_params["destripe_preceded"])
                        else "_destripe"
                    ),
                )
                volumeTranslate_compose(
                    mask,
                    regInfo,
                    AffineMapZXY,
                    T2,
                    zback,
                    z,
                    m,
                    n,
                    padding_z,
                    trans_path,
                    T_flag,
                    f_flag,
                    flip_axes if f_flag else None,
                    device=self.train_params["device"],
                    resample_ratio=self.train_params["resample_ratio"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    z_spacing=self.sample_params["z_spacing"],
                )
                np.save(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "translating_information.npy",
                    ),
                    {
                        "AffineMapZXY": AffineMapZXY,
                        "T2": T2,
                        "zback": zback,
                        "z": z,
                        "m": m,
                        "n": n,
                        "padding_z": padding_z,
                        "T_flag": T_flag,
                        "f_flag": f_flag,
                        "flip_axes": flip_axes,
                    },
                )

        print("\nLocalize sample...")
        print("read in...")
        if not self.train_params["destripe_preceded"]:
            fl = illu_name + ".tif"
        else:
            fl = illu_name + "+RESULT/{}.tif".format(
                self.train_params["destripe_params"]
            )
        f_handle = AICSImage(
            os.path.join(
                save_path, self.sample_params["topillu_ventraldet_data_saving_name"], fl
            )
        )
        illu_front = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        f_handle = AICSImage(
            os.path.join(
                save_path,
                self.sample_params["topillu_dorsaldet_data_saving_name"],
                illu_name
                + "{}{}.tif".format(
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                    "_reg" if require_registration else "",
                ),
            )
        )
        illu_back = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        if not require_registration:
            if f_flag:
                illu_back[:] = np.flip(illu_back, flip_axes)

        cropInfo = self.localizingSample(illu_front.max(0), illu_back.max(0), save_path)
        print(cropInfo)
        if self.train_params["require_precropping"]:
            xs, xe, ys, ye = cropInfo.loc[
                "in summary", ["startX", "endX", "startY", "endY"]
            ].astype(int)
        else:
            xs, xe, ys, ye = None, None, None, None

        if self.train_params["require_precropping"]:
            if len(self.train_params["precropping_params"]) == 0:
                xs, xe, ys, ye = cropInfo.loc[
                    "in summary", ["startX", "endX", "startY", "endY"]
                ].astype(int)
            else:
                if T_flag:
                    ys, ye, xs, xe = self.train_params["precropping_params"]
                else:
                    xs, xe, ys, ye = self.train_params["precropping_params"]
        else:
            xs, xe, ys, ye = None, None, None, None

        fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
        ax1.imshow(illu_front.max(0).T if T_flag else illu_front.max(0))
        if self.train_params["require_precropping"]:
            rect = patches.Rectangle(
                (ys, xs) if (not T_flag) else (xs, ys),
                (ye - ys) if (not T_flag) else (xe - xs),
                (xe - xs) if (not T_flag) else (ye - ys),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax1.add_patch(rect)
        ax1.set_title("ventral det", fontsize=8, pad=1)
        ax1.axis("off")
        ax2.imshow(illu_back.max(0).T if T_flag else illu_back.max(0))
        if self.train_params["require_precropping"]:
            rect = patches.Rectangle(
                (ys, xs) if (not T_flag) else (xs, ys),
                (ye - ys) if (not T_flag) else (xe - xs),
                (xe - xs) if (not T_flag) else (ye - ys),
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            ax2.add_patch(rect)
        ax2.set_title("dorsal det", fontsize=8, pad=1)
        ax2.axis("off")
        plt.show()

        segMask = self.segmentSample(
            illu_front[:, xs:xe, ys:ye][
                :,
                :: self.train_params["resample_ratio"],
                :: self.train_params["resample_ratio"],
            ],
            illu_back[:, xs:xe, ys:ye][
                :,
                :: self.train_params["resample_ratio"],
                :: self.train_params["resample_ratio"],
            ],
            save_path,
        )
        del illu_front, illu_back
        np.save(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "segmentation_det.npy",
            ),
            segMask.transpose(1, 2, 0),
        )

        segMask = np.load(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "segmentation_det.npy",
            )
        ).transpose(2, 0, 1)

        print("\nFor top/left Illu...")
        print("read in...")
        if not self.train_params["destripe_preceded"]:
            if isinstance(
                locals()[
                    "{}_illu_ventral_det_data".format("left" if T_flag else "top")
                ],
                str,
            ):
                top_handle = AICSImage(
                    os.path.join(
                        data_path,
                        locals()[
                            "{}_illu_ventral_det_data".format(
                                "left" if T_flag else "top"
                            )
                        ],
                    )
                )
            else:
                top_handle = AICSImage(
                    locals()[
                        "{}_illu_ventral_det_data".format("left" if T_flag else "top")
                    ]
                )
        else:
            f0 = self.sample_params["topillu_ventraldet_data_saving_name"]
            top_handle = AICSImage(
                os.path.join(
                    save_path,
                    f0,
                    f0 + "+RESULT",
                    self.train_params["destripe_params"] + ".tif",
                )
            )
        rawPlanesTopO = top_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        rawPlanesToptmp = rawPlanesTopO[:, xs:xe, ys:ye]
        _, m_c, n_c = rawPlanesToptmp.shape
        m = len(np.arange(m_c)[:: self.train_params["resample_ratio"]])
        n = len(np.arange(n_c)[:: self.train_params["resample_ratio"]])
        del rawPlanesTopO
        if require_registration:
            bottom_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params[
                        "{}illu_dorsaldet_data_saving_name".format(
                            "bottom"
                            if require_flipping_along_illu_for_dorsaldet
                            else "top"
                        )
                    ],
                    self.sample_params[
                        "{}illu_dorsaldet_data_saving_name".format(
                            "bottom"
                            if require_flipping_along_illu_for_dorsaldet
                            else "top"
                        )
                    ]
                    + "{}_reg.tif".format(
                        ""
                        if (not self.train_params["destripe_preceded"])
                        else "_destripe"
                    ),
                )
            )
        else:
            if not self.train_params["destripe_preceded"]:
                if require_flipping_along_illu_for_dorsaldet:
                    illu_direct = "right" if T_flag else "bottom"
                else:
                    illu_direct = "left" if T_flag else "top"
                if isinstance(
                    locals()["{}_illu_dorsal_det_data".format(illu_direct)],
                    str,
                ):
                    bottom_handle = AICSImage(
                        os.path.join(
                            data_path,
                            locals()["{}_illu_dorsal_det_data".format(illu_direct)],
                        )
                    )
                else:
                    bottom_handle = AICSImage(
                        locals()["{}_illu_dorsal_det_data".format(illu_direct)]
                    )
            else:
                f0 = self.sample_params[
                    "{}illu_dorsaldet_data_saving_name".format(
                        "bottom" if require_flipping_along_illu_for_dorsaldet else "top"
                    )
                ]
                bottom_handle = AICSImage(
                    os.path.join(
                        save_path,
                        f0,
                        f0 + "+RESULT",
                        self.train_params["destripe_params"] + ".tif",
                    )
                )
        rawPlanesBottomO = bottom_handle.get_image_data(
            "ZXY" if T_flag else "ZYX", T=0, C=0
        )
        if not require_registration:
            if f_flag:
                rawPlanesBottomO[:] = np.flip(rawPlanesBottomO, flip_axes)
        m0, n0 = rawPlanesBottomO.shape[-2:]
        rawPlanesBottom = rawPlanesBottomO[:, xs:xe, ys:ye]
        del rawPlanesBottomO
        s = rawPlanesBottom.shape[0]

        if rawPlanesToptmp.shape[0] < rawPlanesBottom.shape[0]:
            rawPlanesTop = np.concatenate(
                (
                    rawPlanesToptmp,
                    np.zeros(
                        (
                            rawPlanesBottom.shape[0] - rawPlanesToptmp.shape[0],
                            rawPlanesBottom.shape[1],
                            rawPlanesBottom.shape[2],
                        ),
                        dtype=np.uint16,
                    ),
                ),
                0,
            )
        else:
            rawPlanesTop = rawPlanesToptmp
        del rawPlanesToptmp

        topF, bottomF, topFBase, bottomFBase = self.extractNSCTF(
            s,
            m,
            n,
            topVol=rawPlanesTop,
            bottomVol=rawPlanesBottom,
            device=self.train_params["device"],
        )

        boundary = self.dualViewFusion(
            topF,
            bottomF,
            segMask,
        )

        boundary = (
            F.interpolate(
                torch.from_numpy(boundary[None, None, :, :]),
                size=(m_c, n_c),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data.numpy()
        )
        boundaryE = np.zeros((m0, n0))
        boundaryE[xs:xe, ys:ye] = boundary
        boundaryE = boundaryE.T
        if xs is not None:
            boundaryE = extendBoundary2(boundaryE, 11)
        if ys is not None:
            boundaryE = extendBoundary2(boundaryE.T, 11).T
        boundaryE = boundaryE.T
        boundaryETop = np.clip(boundaryE, 0, s).astype(np.uint16)

        tifffile.imwrite(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "fusionBoundary_z{}{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                ),
            ),
            boundaryETop.T if T_flag else boundaryETop,
        )
        del topF, bottomF, rawPlanesTop, rawPlanesBottom

        print("\n\nFor bottom/right Illu...")
        print("read in...")
        if not self.train_params["destripe_preceded"]:
            if isinstance(
                locals()[
                    "{}_illu_ventral_det_data".format("right" if T_flag else "bottom")
                ],
                str,
            ):
                top_handle = AICSImage(
                    os.path.join(
                        data_path,
                        locals()[
                            "{}_illu_ventral_det_data".format(
                                "right" if T_flag else "bottom"
                            )
                        ],
                    )
                )
            else:
                top_handle = AICSImage(
                    locals()[
                        "{}_illu_ventral_det_data".format(
                            "right" if T_flag else "bottom"
                        )
                    ]
                )
        else:
            f0 = self.sample_params["bottomillu_ventraldet_data_saving_name"]
            top_handle = AICSImage(
                os.path.join(
                    save_path,
                    f0,
                    f0 + "+RESULT",
                    self.train_params["destripe_params"] + ".tif",
                )
            )
        rawPlanesTopO = top_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        rawPlanesToptmp = rawPlanesTopO[:, xs:xe, ys:ye]
        _, m_c, n_c = rawPlanesToptmp.shape
        m = len(np.arange(m_c)[:: self.train_params["resample_ratio"]])
        n = len(np.arange(n_c)[:: self.train_params["resample_ratio"]])
        del rawPlanesTopO
        if require_registration:
            bottom_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params[
                        "{}illu_dorsaldet_data_saving_name".format(
                            "top"
                            if require_flipping_along_illu_for_dorsaldet
                            else "bottom"
                        )
                    ],
                    self.sample_params[
                        "{}illu_dorsaldet_data_saving_name".format(
                            "top"
                            if require_flipping_along_illu_for_dorsaldet
                            else "bottom"
                        )
                    ]
                    + "{}_reg.tif".format(
                        ""
                        if (not self.train_params["destripe_preceded"])
                        else "_destripe"
                    ),
                )
            )
        else:
            if not self.train_params["destripe_preceded"]:
                if require_flipping_along_illu_for_dorsaldet:
                    illu_direct = "left" if T_flag else "top"
                else:
                    illu_direct = "right" if T_flag else "bottom"
                if isinstance(
                    locals()["{}_illu_dorsal_det_data".format(illu_direct)],
                    str,
                ):
                    bottom_handle = AICSImage(
                        os.path.join(
                            data_path,
                            locals()["{}_illu_dorsal_det_data".format(illu_direct)],
                        )
                    )
                else:
                    bottom_handle = AICSImage(
                        locals()["{}_illu_dorsal_det_data".format(illu_direct)]
                    )
            else:
                f0 = self.sample_params[
                    "{}illu_dorsaldet_data_saving_name".format(
                        "top" if require_flipping_along_illu_for_dorsaldet else "bottom"
                    )
                ]
                bottom_handle = AICSImage(
                    os.path.join(
                        save_path,
                        f0,
                        f0 + "+RESULT",
                        self.train_params["destripe_params"] + ".tif",
                    )
                )
        rawPlanesBottomO = bottom_handle.get_image_data(
            "ZXY" if T_flag else "ZYX", T=0, C=0
        )
        if not require_registration:
            if f_flag:
                rawPlanesBottomO[:] = np.flip(rawPlanesBottomO, flip_axes)
        m0, n0 = rawPlanesBottomO.shape[-2:]
        rawPlanesBottom = rawPlanesBottomO[:, xs:xe, ys:ye]
        del rawPlanesBottomO
        s = rawPlanesBottom.shape[0]
        if rawPlanesToptmp.shape[0] < rawPlanesBottom.shape[0]:
            rawPlanesTop = np.concatenate(
                (
                    rawPlanesToptmp,
                    np.zeros(
                        (
                            rawPlanesBottom.shape[0] - rawPlanesToptmp.shape[0],
                            rawPlanesBottom.shape[1],
                            rawPlanesBottom.shape[2],
                        ),
                        dtype=np.uint16,
                    ),
                ),
                0,
            )
        else:
            rawPlanesTop = rawPlanesToptmp
        del rawPlanesToptmp
        topF, bottomF, topFBase, bottomFBase = self.extractNSCTF(
            s,
            m,
            n,
            topVol=rawPlanesTop,
            bottomVol=rawPlanesBottom,
            device=self.train_params["device"],
        )

        boundary = self.dualViewFusion(
            topF,
            bottomF,
            segMask,
        )

        boundary = (
            F.interpolate(
                torch.from_numpy(boundary[None, None, :, :]),
                size=(m_c, n_c),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data.numpy()
        )

        boundaryE = np.zeros((m0, n0))
        boundaryE[xs:xe, ys:ye] = boundary
        boundaryE = boundaryE.T
        if xs is not None:
            boundaryE = extendBoundary2(boundaryE, 11)
        if ys is not None:
            boundaryE = extendBoundary2(boundaryE.T, 11).T
        boundaryE = boundaryE.T
        boundaryEBottom = np.clip(boundaryE, 0, s).astype(np.uint16)
        tifffile.imwrite(
            os.path.join(
                save_path,
                self.sample_params["bottomillu_ventraldet_data_saving_name"],
                "fusionBoundary_z{}{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                ),
            ),
            boundaryEBottom.T if T_flag else boundaryEBottom,
        )
        del topF, bottomF, rawPlanesTop, rawPlanesBottom

        print("\n\nStitching...")
        print("read in...")
        boundaryEFront = tifffile.imread(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "fusionBoundary_xy{}.tif".format(
                    ""
                    if self.train_params["require_segmentation"]
                    else "_without_segmentation"
                ),
            )
        ).astype(np.float32)

        if require_registration:
            fl = "fusionBoundary_xy{}".format(
                ""
                if self.train_params["require_segmentation"]
                else "_without_segmentation"
            )
            boundaryEBack = np.load(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    fl
                    + "{}_reg.npy".format(
                        ""
                        if (not self.train_params["destripe_preceded"])
                        else "_destripe"
                    ),
                )
            )
        else:
            boundaryEBack = tifffile.imread(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    "fusionBoundary_xy{}.tif".format(
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                )
            ).astype(np.float32)
            if f_flag:
                if require_flipping_along_illu_for_dorsaldet:
                    boundaryEBack = m0 - boundaryEBack
                if require_flipping_along_det_for_dorsaldet:
                    boundaryEBack[:] = np.flip(boundaryEBack, 0)

        boundaryTop = tifffile.imread(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "fusionBoundary_z{}{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                ),
            )
        ).astype(np.float32)
        boundaryBottom = tifffile.imread(
            os.path.join(
                save_path,
                self.sample_params["bottomillu_ventraldet_data_saving_name"],
                "fusionBoundary_z{}{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                ),
            )
        ).astype(np.float32)
        if T_flag:
            boundaryBottom = boundaryBottom.T
            boundaryTop = boundaryTop.T
            if require_registration:
                boundaryEBack = boundaryEBack.swapaxes(1, 2)
        if not self.train_params["destripe_preceded"]:
            fl = illu_name + ".tif"
        else:
            fl = illu_name + "+RESULT/{}.tif".format(
                self.train_params["destripe_params"]
            )

        f_handle = AICSImage(
            os.path.join(
                save_path, self.sample_params["topillu_ventraldet_data_saving_name"], fl
            )
        )
        illu_front = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)

        if require_registration:
            f_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    illu_name
                    + "{}_reg.tif".format(
                        ""
                        if (not self.train_params["destripe_preceded"])
                        else "_destripe"
                    ),
                )
            )
        else:
            f_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    fl,
                )
            )

        illu_back = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        if not require_registration:
            if f_flag:
                illu_back[:] = np.flip(illu_back, flip_axes)
        else:
            if require_flipping_along_illu_for_dorsaldet:
                boundaryEBack = ~boundaryEBack
        s, m0, n0 = illu_back.shape

        require_registration = False

        if require_registration:
            translating_information = np.load(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    "translating_information.npy",
                ),
                allow_pickle=True,
            ).item()
            regInfo = np.load(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    "regInfo{}.npy".format(
                        ""
                        if (not self.train_params["destripe_preceded"])
                        else "_destripe"
                    ),
                ),
                allow_pickle=True,
            ).item()
            invalid_region = (
                volumeTranslate_compose(
                    np.ones((s, m0, n0), dtype=np.float32),
                    regInfo,
                    translating_information["AffineMapZXY"],
                    translating_information["T2"],
                    translating_information["zback"],
                    translating_information["z"],
                    translating_information["m"],
                    translating_information["n"],
                    translating_information["padding_z"],
                    None,
                    translating_information["T_flag"],
                    translating_information["f_flag"],
                    translating_information["flip_axes"],
                    device=self.train_params["device"],
                    resample_ratio=self.train_params["resample_ratio"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    z_spacing=self.sample_params["z_spacing"],
                )
                < 1
            )

            invalid_region[:, xs:xe, ys:ye] = 0

        else:
            invalid_region = np.zeros((s, m0, n0), dtype=bool)

        if save_separate_results:
            reconVol, reconVol_separate = fusionResultFour(
                boundaryTop,
                boundaryBottom,
                boundaryEFront,
                boundaryEBack,
                illu_front,
                illu_back,
                s,
                m0,
                n0,
                save_path,
                self.train_params["device"],
                self.sample_params,
                invalid_region,
                save_separate_results,
                GFr=copy.deepcopy(self.train_params["window_size"]),
            )
        else:
            reconVol = fusionResultFour(
                boundaryTop,
                boundaryBottom,
                boundaryEFront,
                boundaryEBack,
                illu_front,
                illu_back,
                s,
                m0,
                n0,
                save_path,
                self.train_params["device"],
                self.sample_params,
                invalid_region,
                save_separate_results,
                GFr=copy.deepcopy(self.train_params["window_size"]),
            )
        if T_flag:
            result = reconVol.swapaxes(1, 2)
            if save_separate_results:
                result_separate = reconVol_separate.transpose(0, 1, 3, 2)
        else:
            result = reconVol
            if save_separate_results:
                result_separate = reconVol_separate
        del reconVol
        if save_separate_results:
            del reconVol_separate
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, dpi=200)
        ax1.imshow(result.max(0))
        ax1.set_title("result in xy", fontsize=8, pad=1)
        ax1.axis("off")
        ax2.imshow(result.max(1))
        ax2.set_title("result in zy", fontsize=8, pad=1)
        ax2.axis("off")
        ax3.imshow(result.max(2))
        ax3.set_title("result in zx", fontsize=8, pad=1)
        ax3.axis("off")
        plt.show()
        print("Save...")
        tifffile.imwrite(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "quadrupleFusionResult{}{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                ),
            ),
            result,
        )
        if save_separate_results:
            self.save_results(
                os.path.join(
                    save_path, self.sample_params["topillu_ventraldet_data_saving_name"]
                )
                + "/quadrupleFusionResult_separate{}{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                    "" if (not self.train_params["destripe_preceded"]) else "_destripe",
                ),
                result_separate,
            )
            del result_separate
        del illu_front, illu_back
        gc.collect()
        return result

    def save_results(self, save_path, reconVol_separate):
        ijtags = imagej_metadata_tags(
            {"LUTs": [red, orange]},
            ">",
        )
        tifffile.imwrite(
            save_path,
            reconVol_separate,
            byteorder=">",
            imagej=True,
            metadata={"mode": "composite"},
            extratags=ijtags,
            compression="zlib",
            compressionargs={"level": 8},
        )

    def dualViewFusion(self, topF, bottomF, segMask):
        print("to GPU...")
        segMaskGPU = torch.from_numpy(segMask).to(self.train_params["device"])
        topFGPU, bottomFGPU = torch.from_numpy(topF**2).to(
            self.train_params["device"]
        ), torch.from_numpy(bottomF**2).to(self.train_params["device"])
        boundary = EM2DPlus(
            segMaskGPU,
            topFGPU,
            bottomFGPU,
            [
                self.train_params["kernel2d"].shape[0],
                self.train_params["kernel2d"].shape[1],
            ],
            [self.train_params["poly_order"][1], self.train_params["poly_order"][1]],
            self.train_params["kernel2d"],
            self.train_params["n_epochs"],
            device=self.train_params["device"],
            _xy=False,
        )
        del topFGPU, bottomFGPU, segMaskGPU
        return boundary

    def extractNSCTF(self, s, m, n, topVol, bottomVol, device):
        r = self.train_params["resample_ratio"]
        featureExtrac = NSCTdec(levels=[3, 3, 3], device=device)
        topF, bottomF = np.empty((s, m, n), dtype=np.float32), np.empty(
            (s, m, n), dtype=np.float32
        )
        topFBase, bottomFBase = np.empty((s, m, n), dtype=np.float32), np.empty(
            (s, m, n), dtype=np.float32
        )
        tmp0, tmp1 = np.arange(0, s, 10), np.arange(10, s + 10, 10)
        for p, q in tqdm.tqdm(zip(tmp0, tmp1), desc="NSCT: ", total=len(tmp0)):
            topDataFloat, bottomDataFloat = topVol[p:q, :, :].astype(
                np.float32
            ), bottomVol[p:q, :, :].astype(np.float32)
            topDataGPU, bottomDataGPU = torch.from_numpy(
                topDataFloat[:, None, :, :]
            ).to(device), torch.from_numpy(bottomDataFloat[:, None, :, :]).to(device)
            # topDataGPU[:], bottomDataGPU[:] = topDataGPU / Max, bottomDataGPU / Max
            a, b, c = featureExtrac.nsctDec(topDataGPU, r, _forFeatures=True)
            max_filter = nn.MaxPool2d(
                (59, 59), stride=(1, 1), padding=(59 // 2, 59 // 2)
            )
            # c = max_filter(c[None])[0]
            topF[p:q], topFBase[p:q] = (
                c.cpu().detach().numpy(),
                b.cpu().detach().numpy(),
            )
            a[:], b[:], c[:] = featureExtrac.nsctDec(
                bottomDataGPU,
                r,
                _forFeatures=True,
            )
            # c = max_filter(c[None])[0]
            bottomF[p:q], bottomFBase[p:q] = (
                c.cpu().detach().numpy(),
                b.cpu().detach().numpy(),
            )
            del topDataFloat, bottomDataFloat, topDataGPU, bottomDataGPU, a, b, c
        gc.collect()
        return topF, bottomF, topFBase, bottomFBase

    def segmentSample(self, topVoltmp, bottomVol, info_path):
        m, n = topVoltmp.shape[-2:]
        zfront, zback = topVoltmp.shape[0], bottomVol.shape[0]
        if zfront < zback:
            topVol = np.concatenate(
                (topVoltmp, np.zeros((zback - zfront, m, n), dtype=np.uint16)), 0
            )
        else:
            topVol = topVoltmp
        del topVoltmp
        Min, Max = [], []
        th = 0
        allList = [
            value for key, value in self.sample_params.items() if "saving_name" in key
        ]
        for f in allList:
            t = np.load(
                os.path.join(info_path, f, "info.npy"), allow_pickle=True
            ).item()
            Min.append(t["minvol"])
            Max.append(t["maxvol"])
            th += t["thvol"]
        Min = max(Min)
        Max = max(Max)
        th = th / 4
        s = zback
        topSegMask = np.zeros((n, zback, m), dtype=bool)
        bottomSegMask = np.zeros((n, zback, m), dtype=bool)
        for i in tqdm.tqdm(range(zback), desc="watershed: "):
            x_top = topVol[i]
            x_bottom = bottomVol[i]
            th_top = 255 * (morphology.remove_small_objects(x_top > th, 25)).astype(
                np.uint8
            )
            th_bottom = 255 * (
                morphology.remove_small_objects(x_bottom > th, 25)
            ).astype(np.uint8)
            topSegMask[:, i, :] = waterShed(x_top, th_top, Max, Min, m, n).T
            bottomSegMask[:, i, :] = waterShed(x_bottom, th_bottom, Max, Min, m, n).T
        segMask = refineShape(
            topSegMask,
            bottomSegMask,
            None,
            None,
            n,
            s,
            m,
            r=self.train_params["window_size"][1],
            _xy=False,
            max_seg=[-1] * m,
        )
        del topSegMask, bottomSegMask, topVol, bottomVol
        return segMask

    def localizingSample(self, rawPlanes_ventral, rawPlanes_dorsal, info_path):
        cropInfo = pd.DataFrame(
            columns=["startX", "endX", "startY", "endY", "maxv"],
            index=["ventral", "dorsal"],
        )
        for f in ["ventral", "dorsal"]:
            maximumProjection = locals()["rawPlanes_" + f].astype(np.float32)
            maximumProjection = np.log(np.clip(maximumProjection, 1, None))
            m, n = maximumProjection.shape
            allList = [
                value
                for key, value in self.sample_params.items()
                if ("saving_name" in key) and (f in key)
            ]
            th = 0
            maxv = 0
            for l in allList:
                t = np.load(
                    os.path.join(info_path, l, "info.npy"), allow_pickle=True
                ).item()
                th += t["MIP_th"]
                maxv = max(maxv, t["MIP_max"])
            th = th / len(allList)
            thresh = maximumProjection > th
            segMask = morphology.remove_small_objects(thresh, min_size=25)
            d1, d2 = (
                np.where(np.sum(segMask, axis=0) != 0)[0],
                np.where(np.sum(segMask, axis=1) != 0)[0],
            )
            a, b, c, d = (
                max(0, d1[0] - 100),
                min(n, d1[-1] + 100),
                max(0, d2[0] - 100),
                min(m, d2[-1] + 100),
            )
            cropInfo.loc[f, :] = [c, d, a, b, np.exp(maxv)]
        cropInfo.loc["in summary"] = (
            min(cropInfo["startX"]),
            max(cropInfo["endX"]),
            min(cropInfo["startY"]),
            max(cropInfo["endY"]),
            max(cropInfo["maxv"]),
        )
        return cropInfo

    def segMIP(self, maximumProjection, maxv=None, minv=None, th=None):
        m, n = maximumProjection.shape
        if th == None:
            maxv, minv, th = (
                maximumProjection.max(),
                maximumProjection.min(),
                filters.threshold_otsu(maximumProjection),
            )
        thresh = maximumProjection > th
        segMask = morphology.remove_small_objects(thresh, min_size=25)
        d1, d2 = (
            np.where(np.sum(segMask, axis=0) != 0)[0],
            np.where(np.sum(segMask, axis=1) != 0)[0],
        )
        a, b, c, d = (
            max(0, d1[0] - 100),
            min(n, d1[-1] + 100),
            max(0, d2[0] - 100),
            min(m, d2[-1] + 100),
        )
        return a, b, c, d


def fusionResultFour(
    boundaryTop,
    boundaryBottom,
    boundaryFront,
    boundaryBack,
    illu_front,
    illu_back,
    s,
    m,
    n,
    info_path,
    device,
    sample_params,
    invalid_region,
    save_separate_results,
    GFr=49,
):
    zmax = boundaryBack.shape[0]
    decModel = NSCTdec(levels=[3, 3, 3], device=device)

    mask = np.arange(m)[None, :, None]
    if boundaryFront.shape[0] < boundaryBack.shape[0]:
        boundaryFront = np.concatenate(
            (
                boundaryFront,
                np.zeros(
                    (
                        boundaryBack.shape[0] - boundaryFront.shape[0],
                        boundaryFront.shape[1],
                    )
                ),
            ),
            0,
        )
    mask_front = mask > boundaryFront[:, None, :]  ###1是下面，0是上面
    if boundaryBack.ndim == 2:
        mask_back = mask > boundaryBack[:, None, :]
    else:
        mask_back = boundaryBack
    mask_ztop = (
        np.arange(s)[:, None, None] > boundaryTop[None, :, :]
    )  ###1是后面，0是前面
    mask_zbottom = (
        np.arange(s)[:, None, None] > boundaryBottom[None, :, :]
    )  ###1是后面，0是前面

    listPair1 = {"1": "4", "2": "3", "4": "1", "3": "2"}
    reconVol = np.empty(illu_back.shape, dtype=np.uint16)
    if save_separate_results:
        reconVol_separate = np.empty(
            (illu_back.shape[0], 2, illu_back.shape[1], illu_back.shape[2]),
            dtype=np.uint16,
        )
    allList = [
        value
        for key, value in sample_params.items()
        if ("saving_name" in key) and ("dorsal" in key)
    ]
    boundary_mask = np.zeros((s, m, n), dtype=bool)
    volmin = 65535

    for l in allList:
        volmin = min(
            np.load(os.path.join(info_path, l, "info.npy"), allow_pickle=True).item()[
                "minvol"
            ],
            volmin,
        )

    for ii in tqdm.tqdm(range(s), desc="intergrate fusion decision: "):
        if ii < illu_front.shape[0]:
            s1, s2, s3, s4 = (
                copy.deepcopy(illu_front[ii]),
                copy.deepcopy(illu_front[ii]),
                copy.deepcopy(illu_back[ii]),
                copy.deepcopy(illu_back[ii]),
            )
        else:
            s3, s4 = copy.deepcopy(illu_back[ii]), copy.deepcopy(illu_back[ii])
            s1, s2 = np.zeros(s3.shape), np.zeros(s3.shape)

        x = np.zeros((5, 1, m, n), dtype=np.float32)
        x[1, ...] = s1
        x[2, ...] = s2
        x[3, ...] = s3
        x[4, ...] = s4
        xtorch = torch.from_numpy(x).to(device)
        maskList = np.zeros((5, 1, m, n), dtype=bool)
        del x

        List = np.zeros((5, 1, m, n), dtype=np.float32)

        tmp1 = (mask_front[ii] == 0) * (mask_ztop[ii] == 0)  ###top+front
        tmp2 = (mask_front[ii] == 1) * (mask_zbottom[ii] == 0)  ###bottom+front
        tmp3 = (mask_back[ii] == 0) * (mask_ztop[ii] == 1)  ###top+back
        tmp4 = (mask_back[ii] == 1) * (mask_zbottom[ii] == 1)  ###bottom+back

        vnameList = ["1", "2", "3", "4"]

        flag_nsct = 0
        for vname in vnameList:
            maskList[int(vname)] += locals()["tmp" + vname] * (
                ~locals()["tmp" + listPair1[vname]]
            )
            if vnameList.index(vname) < vnameList.index(listPair1[vname]):
                v = locals()["tmp" + vname] * locals()["tmp" + listPair1[vname]]
                if sum(sum(v)):
                    v_labeled, num = measure.label(v, connectivity=2, return_num=True)
                    if flag_nsct == 0:
                        F1, _, _ = decModel.nsctDec(
                            xtorch[int(vname)][None], 1, _forFeatures=True
                        )
                        F2, _, _ = decModel.nsctDec(
                            xtorch[int(listPair1[vname])][None], 1, _forFeatures=True
                        )
                    for vv in range(1, num + 1):
                        v_s = v_labeled == vv
                        if ((F1 - F2).cpu().data.numpy() * v_s).sum() >= 0:
                            maskList[int(vname)] += v_s
                        else:
                            maskList[int(listPair1[vname])] += v_s
                    flag_nsct = 1
        maskList[0] = 1 - maskList[1:].sum(0)
        if maskList[0].sum() > 0:
            if flag_nsct == 0:
                F1, _, _ = decModel.nsctDec(xtorch[1][None], 1, _forFeatures=True)
                F2, _, _ = decModel.nsctDec(xtorch[4][None], 1, _forFeatures=True)
            v_labeled, num = measure.label(maskList[0], connectivity=2, return_num=True)
            for vv in range(1, num + 1):
                v_s = v_labeled == vv
                if ((F1 - F2).cpu().data.numpy() * v_s).sum() >= 0:
                    maskList[1] += v_s
                else:
                    maskList[4] += v_s
        maskList = np.concatenate(
            (maskList[1:2] + maskList[2:3], maskList[3:4] + maskList[4:5]), 0
        )
        boundary_mask[ii] = maskList[0, 0, :, :]

    # np.save("boundary_mask1.npy", boundary_mask)
    _mask_small_tmp = boundary_mask[:, :-2:3, :-2:3]
    _mask_small = np.zeros(
        (s, _mask_small_tmp.shape[1] * 2, _mask_small_tmp.shape[2] * 2), dtype=bool
    )
    _mask_small[:, ::2, ::2] = _mask_small_tmp

    with tqdm.tqdm(
        total=((_mask_small.shape[1] - 1) // 10 + 1)
        * ((_mask_small.shape[2] - 1) // 10 + 1),
        desc="refine along z: ",
        leave=False,
    ) as pbar:
        for i in range((_mask_small.shape[1] - 1) // 10 + 1):
            for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                    morphology.remove_small_objects(
                        _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10],
                        5,
                    )
                )
                pbar.update(1)
    r = copy.deepcopy(_mask_small[:, ::2, ::2])
    _mask_small[:] = 1
    _mask_small[:, ::2, ::2] = r

    with tqdm.tqdm(
        total=((_mask_small.shape[1] - 1) // 10 + 1)
        * ((_mask_small.shape[2] - 1) // 10 + 1),
        desc="refine along z: ",
    ) as pbar:
        for i in range((_mask_small.shape[1] - 1) // 10 + 1):
            for j in range((_mask_small.shape[2] - 1) // 10 + 1):
                _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10] = (
                    morphology.remove_small_holes(
                        _mask_small[:, i * 10 : (i + 1) * 10, j * 10 : (j + 1) * 10],
                        5,
                    )
                )
                pbar.update(1)
    boundary_mask[:, : _mask_small_tmp.shape[1] * 3, : _mask_small_tmp.shape[1] * 3] = (
        np.repeat(np.repeat(_mask_small[:, ::2, ::2], 3, 1), 3, 2)
    )
    boundary_mask[invalid_region] = 1

    # boundary_mask = np.load("boundary_mask2.npy")
    s_f = illu_front.shape[0]
    s_b = illu_back.shape[0]
    if s_f < s_b:
        illu_front = np.concatenate((illu_front, illu_back[-(s_b - s_f) :, :, :]), 0)

    l = np.concatenate(
        (
            np.arange(GFr[0] // 2, 0, -1),
            np.arange(s),
            np.arange(s - GFr[0] // 2, s - GFr[0] + 1, -1),
        ),
        0,
    )

    for ii in tqdm.tqdm(range(2, len(l) - 2), desc="fusion: "):  # topVol.shape[0]
        l_s = l[ii - GFr[0] // 2 : ii + GFr[0] // 2 + 1]

        bottomMask = 1 - boundary_mask[None, l_s, :, :]
        topMask = boundary_mask[None, l_s, :, :]

        ind = ii - GFr[0] // 2

        a, b, c = fusion_perslice(
            illu_front[l_s, :, :].astype(np.float32)[None],
            illu_back[l_s, :, :].astype(np.float32)[None],
            topMask,
            bottomMask,
            GFr,
            device,
        )
        if save_separate_results:
            reconVol[ind], reconVol_separate[ind, 0], reconVol_separate[ind, 1] = (
                a,
                b,
                c,
            )
        else:
            reconVol[ind] = a

    del mask_front, mask_ztop, mask_back, mask_zbottom
    del illu_front, illu_back
    if save_separate_results:
        return reconVol, reconVol_separate
    else:
        return reconVol


def volumeTranslate_compose(
    inputs,
    regInfo,
    AffineMapZXY,
    T2,
    s,
    zmax,
    m,
    n,
    padding_z,
    save_path,
    T_flag,
    f_flag,
    flip_axes,
    device,
    resample_ratio,
    xy_spacing,
    z_spacing,
):
    inputs_dtype = inputs.dtype
    if f_flag:
        inputs[:] = np.flip(inputs, axis=flip_axes)
    if zmax > s:
        rawData = np.concatenate(
            (np.repeat(np.zeros_like(inputs[:1, :, :]), zmax - s, 0), inputs), 0
        )
    else:
        rawData = inputs
    del inputs
    zs, ze, zss, zee = translatingParams(int(round(AffineMapZXY[0])))
    xs, xe, xss, xee = translatingParams(int(round(AffineMapZXY[1])))
    ys, ye, yss, yee = translatingParams(int(round(AffineMapZXY[2])))
    if AffineMapZXY[0] < 0:
        rawData2 = np.concatenate(
            (
                rawData,
                np.repeat(
                    np.zeros_like(rawData[-1:, :, :]), int(np.ceil(-AffineMapZXY[0])), 0
                ),
            ),
            0,
        )
    else:
        rawData2 = rawData
    translatedData = np.zeros(rawData2.shape, dtype=np.float32)
    translatedData[zss:zee, xss:xee, yss:yee] = rawData2[zs:ze, xs:xe, ys:ye]
    del rawData, rawData2
    AffineTransform = regInfo["AffineTransform_float_3_3_inverse"][:, 0]
    afixed = regInfo["fixed_inverse"][:, 0]
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(AffineTransform[:9].astype(np.float64))
    affine.SetTranslation(AffineTransform[9:].astype(np.float64))
    affine.SetCenter(afixed.astype(np.float64))
    A = np.array(affine.GetMatrix()).reshape(3, 3)
    c = np.array(affine.GetCenter())
    t = np.array(affine.GetTranslation())
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = A
    T[0:3, 3] = -np.dot(A, c) + t + c
    T = cupy.asarray(T, dtype=cupy.float32)
    T2 = cupy.asarray(T2, dtype=cupy.float32)
    T_scale = cupy.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.5],
            [0.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=cupy.float32,
    )
    commonData = np.zeros((int(max(padding_z, zmax)), m, n), dtype=inputs_dtype)
    xx, yy = np.meshgrid(
        cupy.arange(commonData.shape[1], dtype=cupy.float32)
        / 2
        * xy_spacing
        * resample_ratio,
        cupy.arange(commonData.shape[2], dtype=cupy.float32)
        / 2
        * xy_spacing
        * resample_ratio,
    )
    xx, yy = xx.T[None], yy.T[None]
    ss = torch.split(torch.arange(commonData.shape[0]), 10)
    for s in tqdm.tqdm(ss, desc="projecting: "):
        start, end = s[0], s[-1] + 1
        start, end = start.item(), end.item()
        Z = (
            z_spacing
            * cupy.ones(xx.shape, dtype=cupy.float32)
            * cupy.arange(start, end, dtype=cupy.float32)[:, None, None]
        )
        X = xx.repeat(end - start, axis=0)
        Y = yy.repeat(end - start, axis=0)
        offset = cupy.ones(Z.shape, dtype=cupy.float32)
        coor = cupy.stack((Z, X, Y, offset))
        del Z, X, Y, offset
        coor_translated = cupy.dot(T_scale, cupy.dot(T, coor.reshape(4, -1)))
        coor_translated /= cupy.array(
            [z_spacing, xy_spacing * resample_ratio, xy_spacing * resample_ratio, 1],
            dtype=cupy.float32,
        )[:, None]
        coor_translated *= cupy.array(
            [z_spacing / xy_spacing, 1, 1, 1], dtype=cupy.float32
        )[:, None]
        coor_translated = cupy.dot(T2, coor_translated)
        coor_translated /= cupy.array(
            [z_spacing / xy_spacing, 1, 1, 1], dtype=cupy.float32
        )[:, None]
        coor_translated = coor_translated[:-1].reshape(
            3, coor.shape[1], coor.shape[2], coor.shape[3]
        )
        del coor
        if coor_translated[0, ...].max() < 0:
            continue
        if coor_translated[0, ...].min() >= translatedData.shape[0]:
            continue
        minn = int(cupy.clip(np.floor(coor_translated[0, ...].min()), 0, None))
        maxx = int(
            cupy.clip(
                cupy.ceil(coor_translated[0, ...].max()), None, translatedData.shape[0]
            )
        )
        smallData = translatedData[minn : maxx + 1, ...]
        if smallData.shape[0] == 1:
            smallData = np.concatenate(
                (smallData, np.zeros((1, m, n), dtype=smallData.dtype)), 0
            )

        coor_translated[0, ...] = coor_translated[0, ...] - minn
        coor_translated = (
            coor_translated
            / cupy.asarray(smallData.shape, dtype=np.float32)[:, None, None, None]
            - 0.5
        ) * 2
        translatedDatasmall = coordinate_mapping(
            smallData,
            coor_translated[[2, 1, 0], ...],
            device=device,
        )
        commonData[start:end, ...] = translatedDatasmall
        del smallData, coor_translated, translatedDatasmall
    if save_path is not None:
        print("save...")
        if T_flag:
            result = commonData.swapaxes(1, 2)
        else:
            result = commonData
        if inputs_dtype == np.uint16:
            tifffile.imwrite(save_path, result)
        else:
            np.save(save_path, result)
        del translatedData, commonData
    else:
        del translatedData
        return commonData


def volumeTranslate2(
    inputs,
    T,
    save_path,
    T_flag,
    device,
    xy_spacing,
    z_spacing,
):
    inputs_dtype = inputs.dtype
    T = cupy.asarray(T, dtype=cupy.float32)
    commonData = np.zeros_like(inputs)
    xx, yy = np.meshgrid(
        cupy.arange(commonData.shape[1], dtype=cupy.float32),
        cupy.arange(commonData.shape[2], dtype=cupy.float32),
    )
    xx, yy = xx.T[None], yy.T[None]
    ss = torch.split(torch.arange(commonData.shape[0]), 10)
    for s in tqdm.tqdm(ss, desc="projecting: "):
        start, end = s[0], s[-1] + 1
        start, end = start.item(), end.item()
        Z = (
            z_spacing
            / xy_spacing
            * cupy.ones(xx.shape, dtype=cupy.float32)
            * cupy.arange(start, end, dtype=cupy.float32)[:, None, None]
        )
        X = xx.repeat(end - start, axis=0)
        Y = yy.repeat(end - start, axis=0)
        offset = cupy.ones(Z.shape, dtype=cupy.float32)
        coor = cupy.stack((Z, X, Y, offset))
        del Z, X, Y, offset
        coor_translated = cupy.dot(T, coor.reshape(4, -1))[:-1].reshape(
            3, coor.shape[1], coor.shape[2], coor.shape[3]
        )
        del coor
        coor_translated /= cupy.array(
            [z_spacing / xy_spacing, 1, 1], dtype=cupy.float32
        )[:, None, None, None]
        if coor_translated[0, ...].max() < 0:
            continue
        if coor_translated[0, ...].min() >= inputs.shape[0]:
            continue
        minn = int(cupy.clip(np.floor(coor_translated[0, ...].min()), 0, None))
        maxx = int(
            cupy.clip(cupy.ceil(coor_translated[0, ...].max()), None, inputs.shape[0])
        )
        smallData = inputs[minn : maxx + 1, ...]
        if smallData.shape[0] == 1:
            smallData = np.concatenate(
                (
                    smallData,
                    np.zeros(
                        (1, smallData.shape[1], smallData.shape[2]),
                        dtype=smallData.dtype,
                    ),
                ),
                0,
            )
        coor_translated[0, ...] = coor_translated[0, ...] - minn
        coor_translated = (
            coor_translated
            / cupy.asarray(smallData.shape, dtype=np.float32)[:, None, None, None]
            - 0.5
        ) * 2
        translatedDatasmall = coordinate_mapping(
            smallData, coor_translated[[2, 1, 0], ...], device=device
        )
        commonData[start:end, ...] = translatedDatasmall
        del smallData, coor_translated, translatedDatasmall
    print("save...")
    print(commonData.shape)
    if T_flag:
        result = commonData.swapaxes(1, 2)
    else:
        result = commonData
    if inputs_dtype == np.uint16:
        tifffile.imwrite(save_path, result)
    else:
        np.save(save_path, result)
    del commonData, result


def volumeTranslate(
    inputs,
    regInfo,
    AffineMapZXY,
    s,
    zmax,
    m,
    n,
    padding_z,
    save_path,
    T_flag,
    f_flag,
    flip_axes,
    device,
    resample_ratio,
    xy_spacing,
    z_spacing,
):
    inputs_dtype = inputs.dtype
    if f_flag:
        inputs[:] = np.flip(inputs, axis=flip_axes)
    if zmax > s:
        rawData = np.concatenate((np.repeat(inputs[:1, :, :], zmax - s, 0), inputs), 0)
    else:
        rawData = inputs
    del inputs
    zs, ze, zss, zee = translatingParams(int(round(AffineMapZXY[0])))
    xs, xe, xss, xee = translatingParams(int(round(AffineMapZXY[1])))
    ys, ye, yss, yee = translatingParams(int(round(AffineMapZXY[2])))
    if AffineMapZXY[0] < 0:
        rawData2 = np.concatenate(
            (
                rawData,
                np.repeat(
                    np.zeros_like(rawData[-1:, :, :]), int(np.ceil(-AffineMapZXY[0])), 0
                ),
            ),
            0,
        )
    else:
        rawData2 = rawData
    translatedData = np.zeros(rawData2.shape, dtype=np.float32)
    translatedData[zss:zee, xss:xee, yss:yee] = rawData2[zs:ze, xs:xe, ys:ye]
    del rawData, rawData2
    AffineTransform = regInfo["AffineTransform_float_3_3_inverse"][:, 0]
    afixed = regInfo["fixed_inverse"][:, 0]
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(AffineTransform[:9].astype(np.float64))
    affine.SetTranslation(AffineTransform[9:].astype(np.float64))
    affine.SetCenter(afixed.astype(np.float64))
    A = np.array(affine.GetMatrix()).reshape(3, 3)
    c = np.array(affine.GetCenter())
    t = np.array(affine.GetTranslation())
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = A
    T[0:3, 3] = -np.dot(A, c) + t + c
    T = cupy.asarray(T, dtype=cupy.float32)
    T_scale = cupy.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.5],
            [0.0, 0.0, 2.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=cupy.float32,
    )
    commonData = np.zeros((int(max(padding_z, zmax)), m, n), dtype=inputs_dtype)
    xx, yy = np.meshgrid(
        cupy.arange(commonData.shape[1], dtype=cupy.float32)
        / 2
        * xy_spacing
        * resample_ratio,
        cupy.arange(commonData.shape[2], dtype=cupy.float32)
        / 2
        * xy_spacing
        * resample_ratio,
    )
    xx, yy = xx.T[None], yy.T[None]
    ss = torch.split(torch.arange(commonData.shape[0]), 10)
    for s in tqdm.tqdm(ss, desc="projecting: "):
        start, end = s[0], s[-1] + 1
        start, end = start.item(), end.item()
        Z = (
            z_spacing
            * cupy.ones(xx.shape, dtype=cupy.float32)
            * cupy.arange(start, end, dtype=cupy.float32)[:, None, None]
        )
        X = xx.repeat(end - start, axis=0)
        Y = yy.repeat(end - start, axis=0)
        offset = cupy.ones(Z.shape, dtype=cupy.float32)
        coor = cupy.stack((Z, X, Y, offset))
        del Z, X, Y, offset
        coor_translated = cupy.dot(T_scale, cupy.dot(T, coor.reshape(4, -1)))[
            :-1
        ].reshape(3, coor.shape[1], coor.shape[2], coor.shape[3])
        del coor
        coor_translated /= cupy.array(
            [z_spacing, xy_spacing * resample_ratio, xy_spacing * resample_ratio],
            dtype=cupy.float32,
        )[:, None, None, None]
        if coor_translated[0, ...].max() < 0:
            continue
        if coor_translated[0, ...].min() >= translatedData.shape[0]:
            continue
        minn = int(cupy.clip(np.floor(coor_translated[0, ...].min()), 0, None))
        maxx = int(
            cupy.clip(
                cupy.ceil(coor_translated[0, ...].max()), None, translatedData.shape[0]
            )
        )
        smallData = translatedData[minn : maxx + 1, ...]
        if smallData.shape[0] == 1:
            smallData = np.concatenate(
                (smallData, np.zeros((1, m, n), dtype=smallData.dtype)), 0
            )
        coor_translated[0, ...] = coor_translated[0, ...] - minn
        coor_translated = (
            coor_translated
            / cupy.asarray(smallData.shape, dtype=np.float32)[:, None, None, None]
            - 0.5
        ) * 2
        translatedDatasmall = coordinate_mapping(
            smallData, coor_translated[[2, 1, 0], ...], device=device
        )
        commonData[start:end, ...] = translatedDatasmall
        del smallData, coor_translated, translatedDatasmall
    print("save...")
    if T_flag:
        result = commonData.swapaxes(1, 2)
    else:
        result = commonData
    if inputs_dtype == np.uint16:
        tifffile.imwrite(save_path, result)
    else:
        np.save(save_path, result)
    del translatedData, commonData


def coordinate_mapping(smallData, coor_translated, device, padding_mode="zeros"):
    coor_translatedCupy = (
        torch.as_tensor(coor_translated, device=device)
        .permute(1, 2, 3, 0)[None]
        .to(device)
    )
    smallDataCupy = torch.from_numpy(smallData.astype(np.float32))[None, None, :, :].to(
        device
    )
    translatedDataCupy = F.grid_sample(
        smallDataCupy,
        coor_translatedCupy,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    translatedDatasmall = translatedDataCupy.squeeze().cpu().data.numpy()
    del translatedDataCupy, smallDataCupy, coor_translatedCupy
    return translatedDatasmall


def boundaryInclude(ft, t, m, n, spacing):
    AffineTransform, afixed = (
        ft["AffineTransform_float_3_3_inverse"][:, 0],
        ft["fixed_inverse"][:, 0],
    )
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(AffineTransform[:9].astype(np.float64))
    affine.SetTranslation(AffineTransform[9:].astype(np.float64))
    affine.SetCenter(afixed.astype(np.float64))
    z = copy.deepcopy(t)
    while 1:
        transformed_point = [
            affine.TransformPoint([float(z), float(j), float(k)])[0]
            for j in [0, m]
            for k in [0, n]
        ]
        zz = min(transformed_point)
        if zz > t + 1:
            break
        z += spacing
    return z


def fineReg(
    respective_view_uint16_pad,
    moving_view_uint16_pad,
    xcs,
    xce,
    ycs,
    yce,
    AffineMapZXY,
    InfoMax,
    save_path,
    destripe_preceded,
    z_spacing,
    xy_spacing,
):
    zs, ze, zss, zee = translatingParams(int(round(AffineMapZXY[0])))
    xs, xe, xss, xee = translatingParams(int(round(AffineMapZXY[1])))
    ys, ye, yss, yee = translatingParams(int(round(AffineMapZXY[2])))
    moving_view_uint16_translated = np.empty(
        moving_view_uint16_pad.shape, dtype=np.uint16
    )
    moving_view_uint16_translated[zss:zee, xss:xee, yss:yee] = moving_view_uint16_pad[
        zs:ze, xs:xe, ys:ye
    ]
    del moving_view_uint16_pad
    respective_view_cropped = respective_view_uint16_pad[:, xcs:xce, ycs:yce]
    moving_view_cropped = moving_view_uint16_translated[:, xcs:xce, ycs:yce]
    del respective_view_uint16_pad, moving_view_uint16_translated
    respective_view_uint8, moving_view_uint8 = np.empty(
        respective_view_cropped.shape, dtype=np.uint8
    ), np.empty(moving_view_cropped.shape, dtype=np.uint8)
    respective_view_uint8[:] = respective_view_cropped / InfoMax * 255
    moving_view_uint8[:] = moving_view_cropped / InfoMax * 255
    del moving_view_cropped, respective_view_cropped
    print("to ANTS...")
    A = np.clip(respective_view_uint8.shape[0] - 200, 0, None) // 2
    staticANTS = ants.from_numpy(
        respective_view_uint8[A : -A if A > 0 else None, ::2, ::2]
    )
    movingANTS = ants.from_numpy(moving_view_uint8[A : -A if A > 0 else None, ::2, ::2])
    movingANTS.set_spacing((z_spacing, xy_spacing * 2, xy_spacing * 2))
    staticANTS.set_spacing((z_spacing, xy_spacing * 2, xy_spacing * 2))
    del moving_view_uint8, respective_view_uint8
    print("registration...")
    regModel = ants.registration(
        staticANTS,
        movingANTS,
        mask=None,
        moving_mask=None,
        type_of_transform="Rigid",
        mask_all_stages=True,
        random_seed=2022,
    )
    shutil.copyfile(regModel["fwdtransforms"][0], os.path.join(save_path, "reg.mat"))
    rfile = scipyio.loadmat(regModel["fwdtransforms"][0])
    rfile_inverse = scipyio.loadmat(regModel["invtransforms"][0])
    del regModel, movingANTS, staticANTS
    return {
        "AffineMapZXY": AffineMapZXY,
        "AffineTransform_float_3_3": np.squeeze(rfile["AffineTransform_float_3_3"]),
        "fixed": np.squeeze(rfile["fixed"]),
        "AffineTransform_float_3_3_inverse": rfile_inverse["AffineTransform_float_3_3"],
        "fixed_inverse": rfile_inverse["fixed"],
        "region_for_reg": np.array([xcs, xce, ycs, yce]),
    }


def translatingParams(x):
    if x == 0:
        xs, xe, xss, xee = None, None, None, None
    elif x > 0:
        xs, xe, xss, xee = x, None, None, -x
    else:
        xs, xe, xss, xee = None, x, -x, None
    return xs, xe, xss, xee


def coarseRegistrationY(front, back, AffineMapZX):
    AffineMapZXY = np.zeros(3)
    AffineMapZXY[:2] = AffineMapZX
    front = front.astype(np.float32)
    back = back.astype(np.float32)
    xs, xe, xss, xee = translatingParams(int(round(AffineMapZX[1])))
    translatedBack = np.zeros(back.shape)
    translatedBack[xss:xee, :] = back[xs:xe, :]
    regModel = ants.registration(
        ants.from_numpy(front),
        ants.from_numpy(translatedBack),
        type_of_transform="Translation",
        random_seed=2022,
    )
    # Y
    AffineMapZXY[2] += scipyio.loadmat(regModel["fwdtransforms"][0])[
        "AffineTransform_float_2_2"
    ][-1, 0]
    # X
    AffineMapZXY[1] += scipyio.loadmat(regModel["fwdtransforms"][0])[
        "AffineTransform_float_2_2"
    ][-2, 0]
    return AffineMapZXY, front, regModel["warpedmovout"].numpy()


def coarseRegistrationZX(yMPfrontO, yMPbackO):
    yMPfrontO = yMPfrontO.astype(np.float32)
    yMPbackO = yMPbackO.astype(np.float32)
    regModel = ants.registration(
        ants.from_numpy(yMPfrontO),
        ants.from_numpy(yMPbackO),
        type_of_transform="Translation",
        random_seed=2022,
    )
    transformedback = regModel["warpedmovout"]
    regModel = ants.registration(
        transformedback,
        ants.from_numpy(yMPbackO),
        type_of_transform="Translation",
        random_seed=2022,
    )
    return scipyio.loadmat(regModel["fwdtransforms"][0])["AffineTransform_float_2_2"][
        -2:
    ][:, 0]
