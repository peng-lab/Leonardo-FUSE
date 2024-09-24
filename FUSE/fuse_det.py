from datetime import datetime


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
from FUSE.blobs_dog import blob_dog
from FUSE.utils import (
    GuidedFilter,
    sgolay2dkernel,
    waterShed,
    refineShape,
    extendBoundary,
    EM2DPlus,
    fusion_perslice,
    extendBoundary2,
    imagej_metadata_tags,
)
from skimage import measure
from FUSE.NSCT import NSCTdec, NSCTrec

# from FUSE.blobs_dog import blob_dog
from typing import Union, Tuple, Optional, List, Dict
import dask
import torch
import os
from aicsimageio import AICSImage
import scipy.ndimage as ndimage
import ants
import scipy.io as scipyio
import open3d as o3d

try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
from skimage import morphology
import shutil
import io
import copy
import SimpleITK as sitk
import tqdm
import gc
import matplotlib.pyplot as plt
import pandas as pd
import dask.array as da

pd.set_option("display.width", 10000)
import matplotlib.patches as patches
import torch.nn as nn
import torch.nn.functional as F
import tifffile
import sys
import h5py


def define_registration_params(
    skip_registration: bool = True,
    skip_transformation: bool = True,
    axial_upsample: int = 1,
    lateral_upsample: int = 1,
):
    kwargs = locals()
    return kwargs


def DoG(input, device):
    points = []
    for ind in tqdm.tqdm(range(input.shape[0]), leave=False, desc="DoG: "):
        tmp = blob_dog(
            input[ind],
            min_sigma=1.8,
            max_sigma=1.8 * 1.6 + 1,
            threshold=0.001,
            th=filters.threshold_otsu(input[ind]),
        )[:, :-1]
        if tmp.shape[0] != 0:
            tmp = tmp.cpu().data.numpy()
            tmp = np.concatenate((ind * np.ones((tmp.shape[0], 1)), tmp), 1).astype(
                np.int32
            )
            points.append(tmp)
    return points


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
        destripe_params: str = "",
        device: str = "cuda",
        registration_params=None,
    ):
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
        if registration_params is None:
            self.registration_params = define_registration_params()
        else:
            self.registration_params = define_registration_params(**registration_params)

    def train_from_params(self, params: dict):
        """Parses training parameters from dictionary"""
        if params["method"] != "detection":
            raise ValueError(f"Invalid method: {params['method']}")
        if params["amount"] not in [2, 4]:
            raise ValueError("Only 2 or 4 images are supported for detection")
        image1 = params["image1"]
        image3 = params["image3"]
        direction1 = params["direction1"]
        direction3 = params["direction3"]

        top_ventral_data = None
        bottom_ventral_data = None
        top_dorsal_data = None
        bottom_dorsal_data = None
        left_ventral_data = None
        right_ventral_data = None
        left_dorsal_data = None
        right_dorsal_data = None

        ventral_data = None
        dorsal_data = None
        left_right = None
        if params["amount"] == 4:
            image2 = params["image2"]
            image4 = params["image4"]
            direction2 = params["direction2"]
            direction4 = params["direction4"]
            if direction1 == "Top" and direction2 == "Bottom":
                top_ventral_data = image1
                bottom_ventral_data = image2
            elif direction1 == "Bottom" and direction2 == "Top":
                top_ventral_data = image2
                bottom_ventral_data = image1
            elif direction1 == "Left" and direction2 == "Right":
                left_ventral_data = image1
                right_ventral_data = image2
            elif direction1 == "Right" and direction2 == "Left":
                left_ventral_data = image2
                right_ventral_data = image1
            else:
                raise ValueError(
                    f"Invalid directions for ventral detection: {direction1}, {direction2}"
                )

            if (
                direction3 not in [direction1, direction2]
                or direction4 not in [direction1, direction2]
                or direction3 == direction4
            ):
                raise ValueError(
                    f"Invalid directions for dorsal detection: {direction3}, {direction4}"
                )

            if direction3 == "Top" and direction4 == "Bottom":
                top_dorsal_data = image3
                bottom_dorsal_data = image4
            elif direction3 == "Bottom" and direction4 == "Top":
                top_dorsal_data = image4
                bottom_dorsal_data = image3
            elif direction3 == "Left" and direction4 == "Right":
                left_dorsal_data = image3
                right_dorsal_data = image4
            elif direction3 == "Right" and direction4 == "Left":
                left_dorsal_data = image4
                right_dorsal_data = image3

        else:
            ventral_data = image1
            dorsal_data = image3
            if (
                direction1 in ["Top", "Bottom"]
                and direction3 in ["Top", "Bottom"]
                and direction1 != direction3
            ):
                left_right = False
            elif (
                direction1 in ["Left", "Right"]
                and direction3 in ["Left", "Right"]
                and direction1 != direction3
            ):
                left_right = True
            else:
                raise ValueError(
                    f"Invalid directions for detection: {direction1}, {direction3}"
                )

        require_registration = params["require_registration"]
        xy_spacing, z_spacing = None, None
        if require_registration:
            z_spacing = params["axial_resolution"]
            xy_spacing = params["lateral_resolution"]
        tmp_path = params["tmp_path"]
        # Create a directory under the intermediate_path
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_dir_path = os.path.join(tmp_path, current_time)
        os.makedirs(new_dir_path, exist_ok=True)

        output_image = self.train(
            require_registration=require_registration,
            require_flipping_along_illu_for_dorsaldet=params["require_flip_illu"],
            require_flipping_along_det_for_dorsaldet=params["require_flip_det"],
            top_illu_ventral_det_data=top_ventral_data,
            bottom_illu_ventral_det_data=bottom_ventral_data,
            top_illu_dorsal_det_data=top_dorsal_data,
            bottom_illu_dorsal_det_data=bottom_dorsal_data,
            left_illu_ventral_det_data=left_ventral_data,
            right_illu_ventral_det_data=right_ventral_data,
            left_illu_dorsal_det_data=left_dorsal_data,
            right_illu_dorsal_det_data=right_dorsal_data,
            ventral_det_data=ventral_data,
            dorsal_det_data=dorsal_data,
            save_path=new_dir_path,
            z_spacing=z_spacing,
            xy_spacing=xy_spacing,
            left_right=left_right,
            # TODO: more parameters?
        )
        if not params["keep_intermediates"]:
            # Clean up the intermediate directory
            shutil.rmtree(new_dir_path)
        return output_image

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
        ventral_det_data: Union[dask.array.core.Array, str] = None,
        dorsal_det_data: Union[dask.array.core.Array, str] = None,
        save_path: str = "",
        save_folder: str = "",
        save_separate_results: bool = False,
        z_spacing: float = None,  # axial
        xy_spacing: float = None,  # lateral
        left_right: bool = None,
    ):
        if require_registration:
            if (z_spacing == None) or (xy_spacing == None):
                print("spacing information is missing.")
                return
        illu_name = "illuFusionResult{}{}".format(
            (
                ""
                if self.train_params["require_segmentation"]
                else "_without_segmentation"
            ),
            self.train_params["destripe_params"],
        )
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return
        save_path = os.path.join(save_path, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        flip_axes = []
        if require_flipping_along_det_for_dorsaldet:
            flip_axes.append(0)
        if require_flipping_along_illu_for_dorsaldet:
            flip_axes.append(1)
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
            det_only_flag = 0
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
            det_only_flag = 0
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
        elif (ventral_det_data is not None) and ((dorsal_det_data is not None)):
            if left_right == None:
                print("left-right marker is missing.")
                return
            if left_right == True:
                T_flag = 1
            else:
                T_flag = 0
            det_only_flag = 1
            if isinstance(ventral_det_data, str):
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    os.path.splitext(ventral_det_data)[0]
                )
            else:
                self.sample_params["topillu_ventraldet_data_saving_name"] = (
                    "ventral_det"
                )
            if isinstance(dorsal_det_data, str):
                self.sample_params["topillu_dorsaldet_data_saving_name"] = (
                    os.path.splitext(dorsal_det_data)[0]
                )
            else:
                self.sample_params["topillu_dorsaldet_data_saving_name"] = "dorsal_det"
        else:
            print("input(s) missing, please check.")
            return

        for k in self.sample_params.keys():
            if "saving_name" in k:
                sub_folder = os.path.join(save_path, self.sample_params[k])
                if not os.path.exists(sub_folder):
                    os.makedirs(sub_folder)

        if not det_only_flag:
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
        else:
            illu_flag_dorsal = 0
            illu_flag_ventral = 0

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
            if (
                not os.path.exists(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                    )
                    + "/regInfo.npy"
                )
            ) or (self.registration_params["skip_registration"] == False):
                print("\nRegister...")
                print("read in...")

                if not det_only_flag:
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
                    respective_view_uint16 = (
                        respective_view_uint16_handle.get_image_data(
                            "ZXY" if T_flag else "ZYX", T=0, C=0
                        )
                    )
                    moving_view_uint16 = moving_view_uint16_handle.get_image_data(
                        "ZXY" if T_flag else "ZYX", T=0, C=0
                    )
                else:
                    if isinstance(ventral_det_data, str):
                        data_handle = AICSImage(
                            os.path.join(
                                data_path,
                                ventral_det_data,
                            )
                        )
                        respective_view_uint16 = data_handle.get_image_data(
                            "ZXY" if T_flag else "ZYX", T=0, C=0
                        )
                    else:
                        if T_flag:
                            tmp = copy.deepcopy(ventral_det_data)
                            respective_view_uint16 = tmp.swapaxes(1, 2)
                            del tmp
                        else:
                            respective_view_uint16 = copy.deepcopy(ventral_det_data)
                    if isinstance(dorsal_det_data, str):
                        data_handle = AICSImage(
                            os.path.join(
                                data_path,
                                dorsal_det_data,
                            )
                        )
                        moving_view_uint16 = data_handle.get_image_data(
                            "ZXY" if T_flag else "ZYX", T=0, C=0
                        )
                    else:
                        if T_flag:
                            tmp = copy.deepcopy(dorsal_det_data)
                            moving_view_uint16 = tmp.swapaxes(1, 2)
                            del tmp
                        else:
                            moving_view_uint16 = copy.deepcopy(dorsal_det_data)

                if isinstance(respective_view_uint16, da.Array):
                    respective_view_uint16 = respective_view_uint16.compute()

                if isinstance(moving_view_uint16, da.Array):
                    moving_view_uint16 = moving_view_uint16.compute()

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
                del respective_view_uint16

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
                    z_spacing=self.sample_params["z_spacing"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    registration_params=self.registration_params,
                )
                del respective_view_uint16_pad, moving_view_uint16_pad
                reg_info.update(
                    {"zfront": s_r, "zback": s_m, "m": m, "n": n, "z": max(s_r, s_m)}
                )
                reg_info.update(self.registration_params)
                np.save(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "regInfo.npy",
                    ),
                    reg_info,
                )

                AffineMapZXY = reg_info["AffineMapZXY"]
                zfront = reg_info["zfront"]
                zback = reg_info["zback"]
                z = reg_info["z"]
                m = reg_info["m"]
                n = reg_info["n"]
                xcs, xce, ycs, yce = reg_info["region_for_reg"]
                padding_z = (
                    boundaryInclude(
                        reg_info,
                        (
                            z + int(np.ceil(-AffineMapZXY[0]))
                            if AffineMapZXY[0] < 0
                            else z
                        )
                        * self.sample_params["z_spacing"],
                        m * self.sample_params["xy_spacing"],
                        n * self.sample_params["xy_spacing"],
                        spacing=self.sample_params["z_spacing"],
                    )
                    / self.sample_params["z_spacing"]
                )
                trans_path = os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    "{}_coarse_reg.tif".format(
                        illu_name
                        if not det_only_flag
                        else self.sample_params["topillu_dorsaldet_data_saving_name"]
                    ),
                )

                volumeTranslate_compose(
                    moving_view_uint16,
                    reg_info["AffineTransform_float_3_3_inverse"],
                    reg_info["fixed_inverse"],
                    AffineMapZXY,
                    None,
                    zback,
                    z,
                    m,
                    n,
                    padding_z,
                    trans_path,
                    T_flag,
                    tuple([]),
                    self.train_params["device"],
                    self.sample_params["xy_spacing"],
                    self.sample_params["z_spacing"],
                )
                del moving_view_uint16
            else:
                print("\nSkip registration...")

            if (
                not os.path.exists(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "regInfo_refine.npy",
                    )
                )
            ) or (self.registration_params["skip_transformation"] == False):
                print("refine registration...")

                if not det_only_flag:
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
                    respective_view_uint16 = (
                        respective_view_uint16_handle.get_image_data(
                            "ZXY" if T_flag else "ZYX", T=0, C=0
                        )
                    )
                    moving_view_uint16 = moving_view_uint16_handle.get_image_data(
                        "ZXY" if T_flag else "ZYX", T=0, C=0
                    )
                else:
                    if isinstance(ventral_det_data, str):
                        data_handle = AICSImage(
                            os.path.join(
                                data_path,
                                ventral_det_data,
                            )
                        )
                        respective_view_uint16 = data_handle.get_image_data(
                            "ZXY" if T_flag else "ZYX", T=0, C=0
                        )
                    else:
                        if T_flag:
                            tmp = copy.deepcopy(ventral_det_data)
                            respective_view_uint16 = tmp.swapaxes(1, 2)
                            del tmp
                        else:
                            respective_view_uint16 = copy.deepcopy(ventral_det_data)

                    if isinstance(respective_view_uint16, da.Array):
                        respective_view_uint16 = respective_view_uint16.compute()

                    moving_view_uint16_handle = AICSImage(
                        os.path.join(
                            save_path,
                            self.sample_params["topillu_dorsaldet_data_saving_name"],
                            self.sample_params["topillu_dorsaldet_data_saving_name"]
                            + "_coarse_reg.tif",
                        )
                    )
                    moving_view_uint16 = moving_view_uint16_handle.get_image_data(
                        "ZXY" if T_flag else "ZYX", T=0, C=0
                    )

                target_points = DoG(
                    respective_view_uint16, device=self.train_params["device"]
                )
                source_points = DoG(
                    moving_view_uint16, device=self.train_params["device"]
                )

                source_points = np.concatenate(source_points, 0)
                target_points = np.concatenate(target_points, 0)

                if source_points.shape[0] > 1e6:
                    d = int(source_points.shape[0] // 1e6)
                else:
                    d = 1

                source_pcd = o3d.geometry.PointCloud()
                source_pcd.points = o3d.utility.Vector3dVector(
                    source_points[::d, :]
                    * np.array(
                        [
                            self.sample_params["z_spacing"],
                            self.sample_params["xy_spacing"],
                            self.sample_params["xy_spacing"],
                        ]
                    )
                )

                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(
                    target_points[::d, :]
                    * np.array(
                        [
                            self.sample_params["z_spacing"],
                            self.sample_params["xy_spacing"],
                            self.sample_params["xy_spacing"],
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
                        "regInfo_refine.npy",
                    ),
                    {
                        "source_points": source_points,
                        "target_points": target_points,
                        "transformation": reg_p2p.transformation,
                    },
                )

                volumeTranslate_compose(
                    moving_view_uint16,
                    None,
                    None,
                    [0, 0, 0],
                    reg_p2p.transformation,
                    moving_view_uint16.shape[0],
                    moving_view_uint16.shape[0],
                    moving_view_uint16.shape[1],
                    moving_view_uint16.shape[2],
                    moving_view_uint16.shape[0],
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_dorsaldet_data_saving_name"],
                        "{}".format(
                            illu_name_
                            if not det_only_flag
                            else self.sample_params[
                                "topillu_dorsaldet_data_saving_name"
                            ]
                        )
                        + "_reg.tif",
                    ),
                    T_flag,
                    tuple([]),
                    self.train_params["device"],
                    self.sample_params["xy_spacing"],
                    self.sample_params["z_spacing"],
                )

                regInfo = np.load(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "regInfo.npy",
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
                        m * self.sample_params["xy_spacing"],
                        n * self.sample_params["xy_spacing"],
                        spacing=self.sample_params["z_spacing"],
                    )
                    / self.sample_params["z_spacing"]
                )
                T2 = reg_p2p.transformation

                np.save(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_ventraldet_data_saving_name"],
                        "translating_information.npy",
                    ),
                    {
                        "AffineTransform_float_3_3_inverse": regInfo[
                            "AffineTransform_float_3_3_inverse"
                        ],
                        "fixed_inverse": regInfo["fixed_inverse"],
                        "AffineMapZXY": AffineMapZXY,
                        "T2": T2,
                        "zback": zback,
                        "z": z,
                        "m": m,
                        "n": n,
                        "padding_z": padding_z,
                        "T_flag": T_flag,
                        "flip_axes": flip_axes,
                    },
                )

                if not det_only_flag:
                    for f, f_name in zip(
                        ["top", "bottom"] if (not T_flag) else ["left", "right"],
                        ["top", "bottom"],
                    ):
                        if isinstance(locals()[f + "_illu_dorsal_det_data"], str):
                            f_handle = AICSImage(
                                os.path.join(
                                    data_path, locals()[f + "_illu_dorsal_det_data"]
                                )
                            )
                        else:
                            f_handle = AICSImage(locals()[f + "_illu_dorsal_det_data"])
                        inputs = f_handle.get_image_data(
                            "ZXY" if T_flag else "ZYX", T=0, C=0
                        )
                        trans_path = os.path.join(
                            save_path,
                            self.sample_params[
                                f_name + "illu_dorsaldet_data_saving_name"
                            ],
                            self.sample_params[
                                f_name + "illu_dorsaldet_data_saving_name"
                            ]
                            + "_reg.tif",
                        )

                        volumeTranslate_compose(
                            inputs,
                            regInfo["AffineTransform_float_3_3_inverse"],
                            regInfo["fixed_inverse"],
                            AffineMapZXY,
                            T2,
                            zback,
                            z,
                            m,
                            n,
                            padding_z,
                            trans_path,
                            T_flag,
                            flip_axes,
                            device=self.train_params["device"],
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
                        fl + "_reg.npy",
                    )
                    volumeTranslate_compose(
                        mask,
                        regInfo["AffineTransform_float_3_3_inverse"],
                        regInfo["fixed_inverse"],
                        AffineMapZXY,
                        T2,
                        zback,
                        z,
                        m,
                        n,
                        padding_z,
                        trans_path,
                        T_flag,
                        flip_axes,
                        device=self.train_params["device"],
                        xy_spacing=self.sample_params["xy_spacing"],
                        z_spacing=self.sample_params["z_spacing"],
                    )

        print("\nLocalize sample...")
        print("read in...")
        fl = illu_name + ".tif"
        if not det_only_flag:
            f_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    fl,
                )
            )
            illu_front = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        else:
            if isinstance(ventral_det_data, str):
                data_handle = AICSImage(
                    os.path.join(
                        data_path,
                        ventral_det_data,
                    )
                )
                illu_front = data_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
            else:
                if T_flag:
                    tmp = copy.deepcopy(ventral_det_data)
                    illu_front = tmp.swapaxes(1, 2)
                    del tmp
                else:
                    illu_front = copy.deepcopy(ventral_det_data)
                if isinstance(illu_front, da.Array):
                    illu_front = illu_front.compute()

        if not det_only_flag:
            f_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    "{}".format(illu_name)
                    + "{}.tif".format(
                        "_reg" if require_registration else "",
                    ),
                )
            )
            illu_back = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        else:
            if require_registration:
                f_handle = AICSImage(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_dorsaldet_data_saving_name"],
                        "{}".format(
                            self.sample_params["topillu_dorsaldet_data_saving_name"]
                        )
                        + "{}.tif".format(
                            "_reg" if require_registration else "",
                        ),
                    )
                )
                illu_back = f_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
            else:
                if isinstance(dorsal_det_data, str):
                    data_handle = AICSImage(
                        os.path.join(
                            data_path,
                            dorsal_det_data,
                        )
                    )
                    illu_back = data_handle.get_image_data(
                        "ZXY" if T_flag else "ZYX", T=0, C=0
                    )
                else:
                    if T_flag:
                        tmp = copy.deepcopy(dorsal_det_data)
                        illu_back = tmp.swapaxes(1, 2)
                        del tmp
                    else:
                        illu_back = copy.deepcopy(dorsal_det_data)
                    if isinstance(illu_back, da.Array):
                        illu_back = illu_back.compute()

        if not require_registration:
            illu_back[:] = np.flip(illu_back, flip_axes)

        cropInfo = self.localizingSample(
            illu_front.max(0), illu_back.max(0), save_path, det_only_flag
        )
        print(cropInfo)

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

        if self.train_params["require_segmentation"]:
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
                det_only_flag,
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
        else:
            segMask = np.ones(
                illu_back[:, xs:xe, ys:ye][
                    :,
                    :: self.train_params["resample_ratio"],
                    :: self.train_params["resample_ratio"],
                ].shape,
                dtype=bool,
            )

        if not det_only_flag:
            print("\nFor top/left Illu...")
        else:
            print("\nEstimate boundary along detection...")

        print("read in...")
        if not det_only_flag:
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
            rawPlanesTopO = top_handle.get_image_data(
                "ZXY" if T_flag else "ZYX", T=0, C=0
            )
        else:
            if isinstance(ventral_det_data, str):
                data_handle = AICSImage(
                    os.path.join(
                        data_path,
                        ventral_det_data,
                    )
                )
                rawPlanesTopO = data_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
            else:
                if T_flag:
                    tmp = copy.deepcopy(ventral_det_data)
                    rawPlanesTopO = tmp.swapaxes(1, 2)
                    del tmp
                else:
                    rawPlanesTopO = copy.deepcopy(ventral_det_data)
            if isinstance(rawPlanesTopO, da.Array):
                rawPlanesTopO = rawPlanesTopO.compute()

        rawPlanesToptmp = rawPlanesTopO[:, xs:xe, ys:ye]
        _, m_c, n_c = rawPlanesToptmp.shape
        m = len(np.arange(m_c)[:: self.train_params["resample_ratio"]])
        n = len(np.arange(n_c)[:: self.train_params["resample_ratio"]])
        del rawPlanesTopO

        if require_registration:
            if not det_only_flag:
                bottom_name = (
                    "bottom" if require_flipping_along_illu_for_dorsaldet else "top"
                )
            else:
                bottom_name = "top"
            bottom_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params[
                        "{}illu_dorsaldet_data_saving_name".format(bottom_name)
                    ],
                    self.sample_params[
                        "{}illu_dorsaldet_data_saving_name".format(bottom_name)
                    ]
                    + "_reg.tif",
                )
            )
        else:
            if not det_only_flag:
                if require_flipping_along_illu_for_dorsaldet:
                    illu_direct = "right" if T_flag else "bottom"
                else:
                    illu_direct = "left" if T_flag else "top"
                _name = locals()["{}_illu_dorsal_det_data".format(illu_direct)]
            else:
                _name = dorsal_det_data
            if isinstance(
                _name,
                str,
            ):
                bottom_handle = AICSImage(
                    os.path.join(
                        data_path,
                        _name,
                    )
                )
            else:
                bottom_handle = AICSImage(_name)

        rawPlanesBottomO = bottom_handle.get_image_data(
            "ZXY" if T_flag else "ZYX", T=0, C=0
        )
        if not require_registration:
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
                "fusionBoundary_z{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                ),
            ),
            boundaryETop.T if T_flag else boundaryETop,
        )
        del topF, bottomF, rawPlanesTop, rawPlanesBottom

        if not det_only_flag:
            print("\n\nFor bottom/right Illu...")
            print("read in...")
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

            rawPlanesTopO = top_handle.get_image_data(
                "ZXY" if T_flag else "ZYX", T=0, C=0
            )
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
                        + "_reg.tif",
                    )
                )
            else:
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
            rawPlanesBottomO = bottom_handle.get_image_data(
                "ZXY" if T_flag else "ZYX", T=0, C=0
            )
            if not require_registration:
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
                    "fusionBoundary_z{}.tif".format(
                        (
                            ""
                            if self.train_params["require_segmentation"]
                            else "_without_segmentation"
                        ),
                    ),
                ),
                boundaryEBottom.T if T_flag else boundaryEBottom,
            )
            del topF, bottomF, rawPlanesTop, rawPlanesBottom

        print("\n\nStitching...")
        print("read in...")
        if not det_only_flag:
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
                        fl + "_reg.npy",
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
                if require_flipping_along_illu_for_dorsaldet:
                    boundaryEBack = m0 - boundaryEBack
                if require_flipping_along_det_for_dorsaldet:
                    boundaryEBack[:] = np.flip(boundaryEBack, 0)

        boundaryTop = tifffile.imread(
            os.path.join(
                save_path,
                self.sample_params["topillu_ventraldet_data_saving_name"],
                "fusionBoundary_z{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                ),
            )
        ).astype(np.float32)
        if not det_only_flag:
            boundaryBottom = tifffile.imread(
                os.path.join(
                    save_path,
                    self.sample_params["bottomillu_ventraldet_data_saving_name"],
                    "fusionBoundary_z{}.tif".format(
                        (
                            ""
                            if self.train_params["require_segmentation"]
                            else "_without_segmentation"
                        ),
                    ),
                )
            ).astype(np.float32)
        if T_flag:
            boundaryTop = boundaryTop.T
            if not det_only_flag:
                boundaryBottom = boundaryBottom.T
                if require_registration:
                    boundaryEBack = boundaryEBack.swapaxes(1, 2)

        if not det_only_flag:
            fl = illu_name + ".tif"

            f_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    fl,
                )
            )
            illu_front = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        else:
            if isinstance(ventral_det_data, str):
                data_handle = AICSImage(
                    os.path.join(
                        data_path,
                        ventral_det_data,
                    )
                )
                illu_front = data_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
            else:
                if T_flag:
                    tmp = copy.deepcopy(ventral_det_data)
                    tmp = tmp.swapaxes(1, 2)
                    illu_front = copy.deepcopy(tmp)
                    del tmp
                else:
                    illu_front = copy.deepcopy(ventral_det_data)
                if isinstance(illu_front, da.Array):
                    illu_front = illu_front.compute()

        if require_registration:
            f_handle = AICSImage(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_dorsaldet_data_saving_name"],
                    "{}".format(
                        illu_name
                        if not det_only_flag
                        else self.sample_params["topillu_dorsaldet_data_saving_name"]
                    )
                    + "_reg.tif",
                )
            )
            illu_back = f_handle.get_image_data("ZXY" if T_flag else "ZYX", T=0, C=0)
        else:
            if not det_only_flag:
                f_handle = AICSImage(
                    os.path.join(
                        save_path,
                        self.sample_params["topillu_dorsaldet_data_saving_name"],
                        fl,
                    )
                )
                illu_back = f_handle.get_image_data(
                    "ZXY" if T_flag else "ZYX", T=0, C=0
                )
            else:
                if isinstance(dorsal_det_data, str):
                    data_handle = AICSImage(
                        os.path.join(
                            data_path,
                            dorsal_det_data,
                        )
                    )
                    illu_back = data_handle.get_image_data(
                        "ZXY" if T_flag else "ZYX", T=0, C=0
                    )
                else:
                    if T_flag:
                        tmp = copy.deepcopy(dorsal_det_data)
                        tmp = tmp.swapaxes(1, 2)
                        illu_back = copy.deepcopy(tmp)
                        del tmp
                    else:
                        illu_back = copy.deepcopy(dorsal_det_data)
                if isinstance(illu_back, da.Array):
                    illu_back = illu_back.compute()

        if not require_registration:
            illu_back[:] = np.flip(illu_back, flip_axes)
        else:
            if require_flipping_along_illu_for_dorsaldet:
                if not det_only_flag:
                    boundaryEBack = ~boundaryEBack
        s, m0, n0 = illu_back.shape

        if require_registration:
            translating_information = np.load(
                os.path.join(
                    save_path,
                    self.sample_params["topillu_ventraldet_data_saving_name"],
                    "translating_information.npy",
                ),
                allow_pickle=True,
            ).item()
            invalid_region = (
                volumeTranslate_compose(
                    np.ones((s, m0, n0), dtype=np.float32),
                    translating_information["AffineTransform_float_3_3_inverse"],
                    translating_information["fixed_inverse"],
                    translating_information["AffineMapZXY"],
                    translating_information["T2"],
                    translating_information["zback"],
                    translating_information["z"],
                    translating_information["m"],
                    translating_information["n"],
                    translating_information["padding_z"],
                    None,
                    translating_information["T_flag"],
                    translating_information["flip_axes"],
                    device=self.train_params["device"],
                    xy_spacing=self.sample_params["xy_spacing"],
                    z_spacing=self.sample_params["z_spacing"],
                )
                < 1
            )
            invalid_region[:, xs:xe, ys:ye] = 0

        else:
            invalid_region = np.zeros((s, m0, n0), dtype=bool)

        if not det_only_flag:
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
        else:
            if save_separate_results:
                reconVol, reconVol_separate = fusionResult(
                    illu_front,
                    illu_back,
                    boundaryTop,
                    self.train_params["device"],
                    save_separate_results,
                    GFr=copy.deepcopy(self.train_params["window_size"]),
                )
            else:
                reconVol = fusionResult(
                    illu_front,
                    illu_back,
                    boundaryTop,
                    self.train_params["device"],
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
                "quadrupleFusionResult{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                ),
            ),
            result,
        )
        if save_separate_results:
            self.save_results(
                os.path.join(
                    save_path, self.sample_params["topillu_ventraldet_data_saving_name"]
                )
                + "/quadrupleFusionResult_separate{}.tif".format(
                    (
                        ""
                        if self.train_params["require_segmentation"]
                        else "_without_segmentation"
                    ),
                ),
                result_separate,
            )
            del result_separate
        del illu_front, illu_back
        gc.collect()
        return result

    def apply(
        self,
        require_registration: bool,
        data_path: str,
        sample_name: str,
        save_path: str,
        save_folder: str,
        ventral_det_data_path: str,
        dorsal_det_data_path: str,
        boundary_path: str,
        translating_information: str,
        z_upsample_ratio: int = 1,
        xy_upsample_ratio: int = 1,
        z_spacing: int = None,
        xy_spacing: int = None,
        skip_apply_registration: bool = False,
        skip_refine_registration: bool = False,
        window_size=[5, 59],
    ):
        if not os.path.exists(save_path):
            print("saving path does not exist.")
            return

        save_path = os.path.join(save_path, save_folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if not os.path.exists(os.path.join(save_path, "high_res")):
            os.makedirs(os.path.join(save_path, "high_res"))

        if require_registration:
            if (z_spacing == None) or (xy_spacing == None):
                print("spacing information is missing.")
                return

        ventral_det_data_handle = h5py.File(
            os.path.join(data_path, sample_name, ventral_det_data_path), "r"
        )
        ventral_det_data = ventral_det_data_handle["X"]
        dorsal_det_data_handle = h5py.File(
            os.path.join(data_path, sample_name, dorsal_det_data_path), "r"
        )
        dorsal_det_data = dorsal_det_data_handle["X"]

        xy_spacing = xy_spacing / xy_upsample_ratio
        z_spacing = z_spacing / z_upsample_ratio

        trans_path = os.path.join(
            save_path, "high_res", os.path.splitext(dorsal_det_data_path)[0] + "_reg.h5"
        )
        if (not skip_apply_registration) or (not os.path.exists(trans_path)):
            print("Apply registration...")
            zback = dorsal_det_data.shape[0]
            zmax = max(ventral_det_data.shape[0], dorsal_det_data.shape[0])
            m = ventral_det_data.shape[1]
            n = ventral_det_data.shape[2]

            translating_information = np.load(
                translating_information, allow_pickle=True
            ).item()

            AffineTransform_float_3_3_inverse = translating_information[
                "AffineTransform_float_3_3_inverse"
            ]
            fixed_inverse = translating_information["fixed_inverse"]
            T2 = translating_information["T2"]
            T_flag = translating_information["T_flag"]
            flip_axes = translating_information["flip_axes"]

            AffineMapZXY = translating_information["AffineMapZXY"] * np.array(
                [z_upsample_ratio, xy_upsample_ratio, xy_upsample_ratio]
            )
            padding_z = translating_information["padding_z"] * z_upsample_ratio

            volumeTranslate_compose(
                dorsal_det_data,
                AffineTransform_float_3_3_inverse,
                fixed_inverse,
                AffineMapZXY,
                T2,
                zback,
                zmax,
                m,
                n,
                padding_z,
                trans_path,
                T_flag,
                flip_axes,
                device=self.train_params["device"],
                xy_spacing=xy_spacing,
                z_spacing=z_spacing,
                high_res=True,
            )
            dorsal_det_data_handle.close()
        else:
            print("Skip coase registration...")

        trans_path = os.path.join(
            save_path,
            "high_res",
            os.path.splitext(dorsal_det_data_path)[0] + "_fine_reg.h5",
        )
        if (not skip_refine_registration) or (not os.path.exists(trans_path)):
            dorsal_det_data_handle = h5py.File(
                os.path.join(
                    save_path,
                    "high_res",
                    os.path.splitext(dorsal_det_data_path)[0] + "_reg.h5",
                ),
                "r",
            )
            dorsal_det_data = dorsal_det_data_handle["X"]

            target_points = DoG(ventral_det_data, device=self.train_params["device"])
            source_points = DoG(dorsal_det_data, device=self.train_params["device"])

            source_points = np.concatenate(source_points, 0)
            target_points = np.concatenate(target_points, 0)

            if source_points.shape[0] > 1e7:
                d = int(source_points.shape[0] // 1e7)
            else:
                d = 1

            source_pcd = o3d.geometry.PointCloud()
            source_pcd.points = o3d.utility.Vector3dVector(
                source_points[::d, :]
                * np.array(
                    [
                        z_spacing,
                        xy_spacing,
                        xy_spacing,
                    ]
                )
            )
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(
                target_points[::d, :]
                * np.array(
                    [
                        z_spacing,
                        xy_spacing,
                        xy_spacing,
                    ]
                )
            )
            print("registration...")
            reg_p2p = o3d.pipelines.registration.registration_icp(
                target_pcd,
                source_pcd,
                3.0,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=True
                ),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=5000),
            )
            np.save(
                os.path.join(
                    save_path,
                    "high_res",
                    "regInfo_refine.npy",
                ),
                {
                    "source_points": source_points,
                    "target_points": target_points,
                    "transformation": reg_p2p.transformation,
                },
            )

            volumeTranslate_compose(
                dorsal_det_data,
                None,
                None,
                [0, 0, 0],
                reg_p2p.transformation,
                dorsal_det_data.shape[0],
                max(dorsal_det_data.shape[0], ventral_det_data.shape[0]),
                dorsal_det_data.shape[1],
                dorsal_det_data.shape[2],
                dorsal_det_data.shape[0],
                trans_path,
                0,
                tuple([]),
                device=self.train_params["device"],
                xy_spacing=xy_spacing,
                z_spacing=z_spacing,
                high_res=True,
            )
            dorsal_det_data_handle.close()
        else:
            print("Skip refine registration...")

        boundary = tifffile.imread(boundary_path)
        dorsal_det_data_handle = h5py.File(
            os.path.join(
                save_path,
                "high_res",
                os.path.splitext(dorsal_det_data_path)[0] + "_fine_reg.h5",
            ),
            "r",
        )
        dorsal_det_data = dorsal_det_data_handle["X"]
        z, x, y = dorsal_det_data.shape
        boundary = (
            z_upsample_ratio
            * F.interpolate(
                torch.from_numpy(boundary[None, None].astype(np.float32)),
                size=(x, y),
                mode="bilinear",
                align_corners=True,
            ).data.numpy()[0, 0]
        )

        reconVol = fusionResult(
            ventral_det_data,
            dorsal_det_data,
            boundary,
            self.train_params["device"],
            False,
            GFr=window_size,
        )

        print("Save...")
        trans_path = os.path.join(save_path, "high_res", "quadrupleFusionResult.h5")
        if os.path.exists(trans_path):
            os.remove(trans_path)
        f = h5py.File(trans_path, "w")
        f.create_dataset("X", data=reconVol)
        f.close()

        del reconVol
        dorsal_det_data_handle.close()
        ventral_det_data_handle.close()

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

    def segmentSample(self, topVoltmp, bottomVol, info_path, det_only_flag):
        if not det_only_flag:
            Min, Max = [], []
            th = 0
            allList = [
                value
                for key, value in self.sample_params.items()
                if "saving_name" in key
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
        else:
            pass

        m, n = topVoltmp.shape[-2:]
        zfront, zback = topVoltmp.shape[0], bottomVol.shape[0]
        if zfront < zback:
            topVol = np.concatenate(
                (topVoltmp, np.zeros((zback - zfront, m, n), dtype=np.uint16)), 0
            )
        else:
            topVol = copy.deepcopy(topVoltmp)
        del topVoltmp
        s = zback
        topSegMask = np.zeros((n, zback, m), dtype=bool)
        bottomSegMask = np.zeros((n, zback, m), dtype=bool)
        for i in tqdm.tqdm(range(zback), desc="watershed: "):
            x_top = topVol[i]
            x_bottom = bottomVol[i]
            if det_only_flag:
                th = filters.threshold_otsu(x_top + 0.0 + x_bottom) / 2
                Min = max(x_top.min(), x_bottom.min())
                Max = max(x_top.max(), x_bottom.max())
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

    def localizingSample(
        self, rawPlanes_ventral, rawPlanes_dorsal, info_path, det_only_flag
    ):
        cropInfo = pd.DataFrame(
            columns=["startX", "endX", "startY", "endY", "maxv"],
            index=["ventral", "dorsal"],
        )
        for f in ["ventral", "dorsal"]:
            maximumProjection = locals()["rawPlanes_" + f].astype(np.float32)
            maximumProjection = np.log(np.clip(maximumProjection, 1, None))
            m, n = maximumProjection.shape
            if not det_only_flag:
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
            else:
                thresh = maximumProjection > filters.threshold_otsu(maximumProjection)
                maxv = np.log(max(rawPlanes_ventral.max(), rawPlanes_dorsal.max()))
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


def fusionResult(
    topVol,
    bottomVol,
    boundary,
    device,
    save_separate_results,
    GFr=[5, 49],
):
    s, m, n = bottomVol.shape
    boundary = boundary[None, None, :, :]
    if save_separate_results:
        reconVol_separate = np.empty((s, 2, m, n), dtype=np.uint16)
    mask = np.arange(s)[None, :, None, None]
    GFr[1] = GFr[1] // 4 * 2 + 1
    boundary = mask > boundary

    l = np.concatenate(
        (
            np.zeros(GFr[0] // 2, dtype=np.int32),
            np.arange(s),
            (s - 1) * np.ones(s - GFr[0] // 2 - (s - GFr[0] + 1), dtype=np.int32),
        ),
        0,
    )
    recon = np.zeros(bottomVol.shape, dtype=np.uint16)

    for ii in tqdm.tqdm(
        range(GFr[0] // 2, len(l) - GFr[0] // 2), desc="fusion: "
    ):  # topVol.shape[0]
        l_s = l[ii - GFr[0] // 2 : ii + GFr[0] // 2 + 1]

        bottomMask = torch.from_numpy(boundary[:, l_s, :, :]).to(device).to(torch.float)
        topMask = torch.from_numpy((~boundary[:, l_s, :, :])).to(device).to(torch.float)

        ind = ii - GFr[0] // 2

        topVol_tmp = np.zeros((len(l_s), m, n), dtype=np.float32)
        bottomVol_tmp = np.zeros((len(l_s), m, n), dtype=np.float32)

        indd = np.where(l_s < topVol.shape[0])[0]
        tmp = l_s[indd]
        b, c = np.unique(tmp, return_counts=True)

        topVol_tmp[indd, :, :] = topVol[b, :, :].repeat(c, 0)
        b, c = np.unique(l_s, return_counts=True)
        bottomVol_tmp[:] = bottomVol[b, :, :].repeat(c, 0)

        # topVol[l_s, :, :].astype(np.float32)[None]

        a, b, c = fusion_perslice(
            topVol_tmp[None],
            bottomVol_tmp[None],
            topMask,
            bottomMask,
            GFr,
            device,
        )
        if save_separate_results:
            recon[ind], reconVol_separate[ind, 0], reconVol_separate[ind, 1] = a, b, c
        else:
            recon[ind] = a
    if save_separate_results:
        return recon, reconVol_separate
    else:
        return recon


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
    AffineTransform_float_3_3_inverse,
    fixed_inverse,
    AffineMapZXY,
    T2,
    s,
    zmax,
    m,
    n,
    padding_z,
    save_path,
    T_flag,
    flip_axes,
    device,
    xy_spacing,
    z_spacing,
    high_res=False,
):
    zs, ze, zss, zee = translatingParams(int(round(AffineMapZXY[0])))
    xs, xe, xss, xee = translatingParams(int(round(AffineMapZXY[1])))
    ys, ye, yss, yee = translatingParams(int(round(AffineMapZXY[2])))

    if AffineTransform_float_3_3_inverse is not None:
        AffineTransform = AffineTransform_float_3_3_inverse[:, 0]
        afixed = fixed_inverse[:, 0]
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
        T = torch.from_numpy(T.astype(np.float32)).to(device)
    else:
        T = None
    if T2 is not None:
        T2 = torch.from_numpy(T2.astype(np.float32)).to(device)

    inputs_dtype = inputs.dtype
    if high_res and T_flag:
        (m, n) = (n, m)
        T_plus_flag = 1
    else:
        T_plus_flag = 0

    z_list = np.arange(s)
    if np.isin(0, flip_axes):
        z_list = z_list[::-1]
        flip_z = True
    else:
        flip_z = False

    if zmax > s:
        z_list = np.concatenate((np.ones(zmax - s, dtype=np.int32) * -1, z_list))

    if AffineMapZXY[0] < 0:
        z_list = np.concatenate(
            (z_list, np.ones(int(np.ceil(-AffineMapZXY[0])), dtype=np.int32) * -1)
        )

    if np.ceil(padding_z) > len(z_list):
        z_list = np.concatenate(
            (
                z_list,
                np.ones(int(np.ceil(padding_z) - len(z_list)), dtype=np.int32) * -1,
            )
        )

    tmp = np.ones_like(z_list) * -1
    tmp[zss:zee] = z_list[zs:ze]
    z_list = copy.deepcopy(tmp)
    del tmp

    commonData = np.zeros(
        (len(z_list), m if not T_flag else n, n if not T_flag else m),
        dtype=inputs_dtype,
    )

    yy, xx = torch.meshgrid(
        torch.arange(commonData.shape[2]).to(torch.float).to(device) * xy_spacing,
        torch.arange(commonData.shape[1]).to(torch.float).to(device) * xy_spacing,
        indexing="ij",
    )
    xx, yy = xx.T[None], yy.T[None]
    ss = torch.split(torch.arange(commonData.shape[0]), 10)

    for s in tqdm.tqdm(ss, desc="projecting: "):
        start, end = s[0], s[-1] + 1
        start, end = start.item(), end.item()
        Z = (
            z_spacing
            * torch.ones_like(xx)
            * torch.arange(start, end, dtype=torch.float, device=device)[:, None, None]
        )
        X = xx.repeat(end - start, 1, 1)
        Y = yy.repeat(end - start, 1, 1)
        offset = torch.ones_like(Z)
        coor = torch.stack((Z, X, Y, offset))
        del Z, X, Y, offset
        if T is not None:
            coor_translated = torch.matmul(T, coor.reshape(4, -1))
        else:
            coor_translated = copy.deepcopy(coor.reshape(4, -1))
        if T2 is not None:
            coor_translated = torch.matmul(T2, coor_translated)
        norm = torch.from_numpy(
            np.array([z_spacing, xy_spacing, xy_spacing, 1]).astype(np.float32)
        ).to(device)[:, None]
        coor_translated /= norm

        coor_translated = coor_translated[:-1].reshape(
            3, coor.shape[1], coor.shape[2], coor.shape[3]
        )
        del coor
        if coor_translated[0, ...].max() < 0:
            continue
        if coor_translated[0, ...].min() >= len(z_list):
            continue
        minn = int(torch.clip(torch.floor(coor_translated[0, ...].min()), 0, None))
        maxx = int(
            torch.clip(torch.ceil(coor_translated[0, ...].max()), None, len(z_list))
        )

        z_list_small = z_list[minn : maxx + 1]
        smallData = np.zeros((len(z_list_small), m, n), dtype=inputs_dtype)
        ind = np.where(z_list_small != -1)[0]
        if flip_z:
            tmp = inputs[np.flip(z_list_small[ind]), :, :][::-1, :, :]
        else:
            tmp = inputs[z_list_small[ind], :, :]
        if T_plus_flag:
            tmp = tmp.swapaxes(1, 2)
        smallData[ind, :, :] = tmp
        if np.isin(1, flip_axes):
            smallData[:] = np.flip(smallData, 1)
        del tmp
        smallData_translate = np.zeros_like(smallData)
        smallData_translate[:, xss:xee, yss:yee] = smallData[:, xs:xe, ys:ye]
        del smallData
        if smallData_translate.shape[0] == 1:
            smallData_translate = np.concatenate(
                (
                    smallData_translate,
                    np.zeros((1, m, n), dtype=smallData_translate.dtype),
                ),
                0,
            )

        coor_translated[0, ...] = coor_translated[0, ...] - minn
        coor_translated = (
            coor_translated
            / torch.Tensor(smallData_translate.shape).to(device)[:, None, None, None]
            - 0.5
        ) * 2
        translatedDatasmall = coordinate_mapping(
            smallData_translate,
            coor_translated[[2, 1, 0], ...],
            device=device,
        )
        if translatedDatasmall.ndim == 2:
            translatedDatasmall = translatedDatasmall[None]
        if not T_flag:
            commonData[start:end, ...] = translatedDatasmall
        else:
            commonData[start:end, ...] = translatedDatasmall.swapaxes(1, 2)
        del smallData_translate, coor_translated, translatedDatasmall

    if save_path is not None:
        print("save...")
        if ".tif" == os.path.splitext(save_path)[1]:
            tifffile.imwrite(save_path, commonData)
        elif ".npy" == os.path.splitext(save_path)[1]:
            np.save(save_path, commonData)
        else:
            if os.path.exists(save_path):
                os.remove(save_path)
            f = h5py.File(save_path, "w")
            f.create_dataset("X", data=commonData)
            f.close()
        del commonData
    else:
        return commonData


def coordinate_mapping(smallData, coor_translated, device, padding_mode="zeros"):
    smallDataCupy = torch.from_numpy(smallData.astype(np.float32))[None, None, :, :].to(
        device
    )
    translatedDataCupy = F.grid_sample(
        smallDataCupy,
        coor_translated.permute(1, 2, 3, 0)[None],
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )
    translatedDatasmall = translatedDataCupy.squeeze().cpu().data.numpy()
    del translatedDataCupy, smallDataCupy
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
    z_spacing,
    xy_spacing,
    registration_params,
):
    zs, ze, zss, zee = translatingParams(int(round(AffineMapZXY[0])))
    xs, xe, xss, xee = translatingParams(int(round(AffineMapZXY[1])))
    ys, ye, yss, yee = translatingParams(int(round(AffineMapZXY[2])))
    moving_view_uint16_translated = np.empty(
        moving_view_uint16_pad.shape, dtype=np.uint16
    )
    moving_view_uint16_translated[zss:zee, xss:xee, yss:yee] = moving_view_uint16_pad[
        zs:ze,
        xs:xe,
        ys:ye,
    ]
    del moving_view_uint16_pad
    respective_view_cropped = respective_view_uint16_pad[:, xcs:xce, ycs:yce]
    moving_view_cropped = moving_view_uint16_translated[:, xcs:xce, ycs:yce]
    del respective_view_uint16_pad, moving_view_uint16_translated
    respective_view_uint8 = np.empty(respective_view_cropped.shape, dtype=np.uint8)
    moving_view_uint8 = np.empty(moving_view_cropped.shape, dtype=np.uint8)
    respective_view_uint8[:] = respective_view_cropped / InfoMax * 255
    moving_view_uint8[:] = moving_view_cropped / InfoMax * 255

    size = (
        sys.getsizeof(respective_view_uint8)
        / registration_params["axial_upsample"]
        / (registration_params["lateral_upsample"]) ** 2
    )
    if size < 209715344:
        s = 0
        e = None
    else:
        r = 209715344 // sys.getsizeof(
            np.ones(
                int(
                    moving_view_uint8[0].size
                    / (registration_params["lateral_upsample"]) ** 2
                ),
                dtype=np.uint8,
            )
        )
        s = (moving_view_uint8.shape[0] - r) // 2
        e = -s
        print("only [{}, {}] slices will be used for registration...".format(s, e))

    del moving_view_cropped, respective_view_cropped
    print("to ANTS...")
    staticANTS = ants.from_numpy(
        respective_view_uint8[
            s : e : registration_params["axial_upsample"],
            :: registration_params["lateral_upsample"],
            :: registration_params["lateral_upsample"],
        ]
    )
    movingANTS = ants.from_numpy(
        moving_view_uint8[
            s : e : registration_params["axial_upsample"],
            :: registration_params["lateral_upsample"],
            :: registration_params["lateral_upsample"],
        ]
    )
    movingANTS.set_spacing(
        (
            z_spacing * registration_params["axial_upsample"],
            xy_spacing * registration_params["lateral_upsample"],
            xy_spacing * registration_params["lateral_upsample"],
        )
    )
    staticANTS.set_spacing(
        (
            z_spacing * registration_params["axial_upsample"],
            xy_spacing * registration_params["lateral_upsample"],
            xy_spacing * registration_params["lateral_upsample"],
        )
    )
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
    translatedBack = np.zeros_like(back)
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
