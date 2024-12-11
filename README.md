# Leonardo-FUSE (LSFM Fusion in Python)

[![Build Status](https://github.com/peng-lab/Leonardo-FUSE/workflows/Build%20Main/badge.svg)](https://github.com/peng-lab/Leonardo-FUSE/actions)

A Python implementation of LSFM fusion method (former BigFUSE)

As a part of Leonardo package, this is fusion method is to fuse LSFM datasets illuminated via opposite illumination lenses and/or detected via opposing detecion lenses. 

---
## Quick Start
### Fusing two datasets illuminated with opposite light sources
#### Use as Python API
(1) Provide two filenames

suppose we have two to-be-processed volumes "A.tif" and "B.tif" saved in folder `sample_name` under the path `data_path`.
Meanwhile fusion result, together with the intermediate results, will be saved in folder `save_folder` under the path `save_path`.

if "A.tif" and "B.tif" are illuminated from top and bottom respectively (in the image space):
```python
from FUSE import FUSE_illu

exe = FUSE_illu()   ###to run with default parameters for training
out = exe.train(data_path = data_path,
                sample_name = sample_name,
                top_illu_data = "A.tif",
                bottom_illu_data = "B.tif",
                save_path = save_path,
                save_folder = save_folder,
                save_separate_results = True,
                sparse_sample = False,
                cam_pos = "front",
                )
```
where `save_separate_results` means whether to save the separated results before stitching (by default False), `sparse_sample` means whether the specimen is sparse (by default False), and `cam_pos` means whether the detection device is in the front or back in the image space (by default "front"). Two folders, named The fusion result, named "A" ad "B" will be created under `save_folder` to keep intermediate results, whereas the fusion result, named "illuFusionResult.tif" will be saved in foldedr "A", i.e., named after the volume illuminated from the top. Fusion result will also be returned as `out`.

otherwise,  "A.tif" and "B.tif" can be illuminated from left and right respectively:
```python
out = exe.train(data_path = data_path,
                sample_name = sample_name,
                left_illu_data = "A.tif",
                right_illu_data = "B.tif",
                save_path = save_path,
                save_folder = save_folder,
                )
```
Alternatively, FUSE-illu can be initialized with user-defined training parameters, a full list of input arguments in `__init__` is here:
```
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
```

(2) Provide two arrays, i.e., the stacks to be processed, and necessary parameters (suitable for use in napari)
suppose we have two to-be-processed volumes `img_arr1` (illuminated from the top) and `img_arr2` (illuminated from the bottom) that have been read in as a np.ndarray or dask.array.core.Array
```python
from FUSE import FUSE_illu

exe = FUSE_illu()   ###to run with default parameters for training
out = exe.train(data_path = data_path,
                sample_name = sample_name,
                top_illu_data = img_arr1,
                bottom_illu_data = img_arr2,
                save_path = save_path,
                save_folder = save_folder,
                save_separate_results = True,
                sparse_sample = False,
                cam_pos = "front",
                )
```
to save results, two folders "top_illu" and "bottom_illu" will be created under `save_folder`, and the fusion result "illuFusionResult.tif" will be kept in folder "top_illu".

same for left-right illumination orientations, simply use arguments `left_illu_data` and `right_illu_data`, instead of `top_illu_data` and `bottom_illu_data`.

#### Run from command line for batch processing
suppose we have two to-be-processed volumes saved in /path/to/my/sample_name/image1.tiff (from the top) and /path/to/my/sample_name/image2.tiff (from the bottom), and we'd like to save the result in /save_path/save_folder:
```bash
fuse_illu --data_path /path/to/my \
          --sample_name sample_name \
          --top_illu_data image1.tiff \
          --bottom_illu_data image2.tiff \
          --save_path /save_path \
          --save_folder save_folder
```
in addition, all the training parameters can be changed in command line:
```bash
fuse_illu --data_path /path/to/my \
          --sample_name sample_name \
          --top_illu_data image1.tiff \
          --bottom_illu_data image2.tiff \
          --save_path /save_path \
          --save_folder save_folder\
          --window_size 5,59 \
          --poly_order 2,2 
```
a full list of changeable args:
```
usage: run_fuse_illu --data_path
                     --sample_name
                     --save_path
                     --save_folder
                     [--require_precropping "True"]
                     [--precropping_params []]
                     [--resample_ratio 2]
                     [--window_size [5, 59]]
                     [--poly_order [2, 2]]
                     [--n_epochs 50]
                     [--require_segmentation "True"]
                     [--device "cpu"]
                     [--top_illu_data None]
                     [--bottom_illu_data None]
                     [--left_illu_data None]
                     [--right_illu_data None]
                     [--camera_position ""]
                     [--cam_pos "front"]
                     [--sparse_sample "False"]
                     [--save_separate_results "False"]
```









### Fusing four datasets with dual-sided illumination and dual-sided detection
#### Use as Python API
(1) Provide two filenames

suppose we have four to-be-processed volumes "A.tif", "B.tif", "C.tif" and "D.tif" saved in folder `sample_name` under the path `data_path`.
Meanwhile fusion result, together with the intermediate results, will be saved in folder `save_folder` under the path `save_path`.

if "A.tif", "B.tif", "C.tif" and "D.tif" are top illuminated+ventral detected, bottom illuminated+ventral detected, top illuminated+dorsal detected, bottom illuminated+dorsal detected (in the image space), respectively:
```python
from FUSE import FUSE_det

exe = FUSE_det()   ###to run with default parameters for training
out = exe.train(data_path = data_path,
                sample_name = sample_name,
                require_registration = require_registration,
                require_flipping_along_illu_for_dorsaldet = require_flipping_along_illu_for_dorsaldet,
                require_flipping_along_det_for_dorsaldet = require_flipping_along_det_for_dorsaldet,
                top_illu_ventral_det_data = "A.tif",
                bottom_illu_ventral_det_data = "B.tif",
                top_illu_dorsal_det_data = "C.tif",
                bottom_illu_dorsal_det_data =  "D.tif",
                save_path = save_path,
                save_folder = save_folder,
                save_separate_results = False,
                sparse_sample = False,
                z_spacing = z_spacing,
                xy_spacing = xy_spacing,
                )

```
where `require_registration` means whether registration for the two detection devices is needed. By default, we assume datasets with different illumination sources but the same detection device are well registered in advance. `require_flipping_along_illu_for_dorsaldet` and `require_flipping_along_det_for_dorsaldet` mean in order to put inputs in a common space, whether flipping along illumination and detection are needed, respectively. Flipping only applies to datasets with detection device at the back. `save_separate_results` and `sparse_sample` are same as in FUSE_illu. If `require_registration` is True, `z_spacing` and  `z_spacing`, axial resolution and lateral resolution, respectively, are mandatory for registration. Otherwise, they are optional.

Four folders "A", "B", "C" and "D" will be created under `save_foldedr` to keep intermediate results, and the fusion result, "quadrupleFusionResult.tif" will be saved under folder "A".

otherwise,  illmination can be of hotizontal orientations (in image space):
```python
out = exe.train(data_path = data_path,
                sample_name = sample_name,
                require_registration = require_registration,
                require_flipping_along_illu_for_dorsaldet = require_flipping_along_illu_for_dorsaldet,
                require_flipping_along_det_for_dorsaldet = require_flipping_along_det_for_dorsaldet,
                left_illu_ventral_det_data = "A.tif",
                right_illu_ventral_det_data = "B.tif",
                left_illu_dorsal_det_data = "C.tif",
                right_illu_dorsal_det_data =  "D.tif",
                save_path = save_path,
                save_folder = save_folder,
                )
```
Alternatively, FUSE-det can be initialized with user-defined training parameters, a full list of input arguments in `__init__` is here:
```
require_precropping: bool = True,
precropping_params: list[int, int, int, int] = [],
resample_ratio: int = 2,
window_size: list[int, int] = [5, 59],
poly_order: list[int, int] = [2, 2],
n_epochs: int = 50,
require_segmentation: bool = True,
device: str = "cpu",
```

(2) Provide four arrays, i.e., the stacks to be processed, and necessary parameters (suitable for use in napari)
suppose we have four to-be-processed volumes `img_arr1` (top illuminated+ventral detected) and `img_arr2` (bottom illuminated+ventral detected), `img_arr3` (top illuminated+dorsal detected) and `img_arr4` (bottom illuminated+dorsal detected). All of them have been read in as a np.ndarray or dask.array.core.Array:
```python
from FUSE import FUSE_det

exe = FUSE_det()   ###to run with default parameters for training
out = exe.train(data_path = data_path,
                sample_name = sample_name,
                require_registration = require_registration,
                require_flipping_along_illu_for_dorsaldet = require_flipping_along_illu_for_dorsaldet,
                require_flipping_along_det_for_dorsaldet = require_flipping_along_det_for_dorsaldet,
                left_illu_ventral_det_data = img_arr1,
                right_illu_ventral_det_data = img_arr2,
                left_illu_dorsal_det_data = img_arr3,
                right_illu_dorsal_det_data = img_arr4,
                save_path = save_path,
                save_folder = save_folder,
                )
```
to save results, four folders "top_illu+ventral_det", "bottom_illu+ventral_det", "top_illu+dorsal_det", and "bottom_illu+dorsal_det" will be created under `save_folder`, and the fusion result "quadrupleFusionResult.tif" will be kept in folder "top_illu+ventral_det".

same for left-right illumination orientations, simply use arguments `left_illu_ventral_det_data`, `right_illu_ventral_det_data`, `left_illu_dorsal_det_data` and `right_illu_dorsal_det_data`.

#### Run from command line for batch processing
suppose we have four to-be-processed volumes saved in /path/to/my/sample_name/image1.tiff (top illuminated+ventral detected), /path/to/my/sample_name/image2.tiff (bottom illuminated+ventral detected), /path/to/my/sample_name/image3.tiff (top illuminated+dorsal detected), and /path/to/my/sample_name/image4.tiff (bottom illuminated+dorsal detected). We'd like to save the result in /save_path/save_folder:
```bash
bigfuse_det --data_path /path/to/my \
            --sample_name sample_name \
            --require_registration False\
            --require_flipping_along_illu_for_dorsaldet True \
            --require_flipping_along_det_for_dorsaldet False \
            --top_illu_ventral_det_data image1.tiff \
            --bottom_illu_ventral_det_data image2.tiff \
            --top_illu_dorsal_det_data image3.tiff \
            --bottom_illu_dorsal_det_data image4.tiff \
            --save_path /save_path \
            --save_folder save_folder
```
in addition, all the training parameters can be changed in command line. A full list of changeable args:
```
usage: run_fuse_det --data_path
                    --sample_name
                    --save_path
                    --save_folder
                    --require_registration
                    --require_flipping_along_illu_for_dorsaldet
                    --require_flipping_along_det_for_dorsaldet
                    [--require_precropping "True"]
                    [--precropping_params []]
                    [--resample_ratio 2]
                    [--window_size [5, 59]]
                    [--poly_order [2, 2]]
                    [--n_epochs 50]
                    [--require_segmentation "True"]
                    [--skip_illuFusion "False"]
                    [--destripe_preceded  "False"]
                    [--destripe_params None]
                    [--device "cpu"]
                    [--sparse_sample "False"]
                    [--top_illu_ventral_det_data None]
                    [--bottom_illu_ventral_det_data None]
                    [--top_illu_dorsal_det_data None]
                    [--bottom_illu_dorsal_det_data None]
                    [--left_illu_ventral_det_data None]
                    [--right_illu_ventral_det_data None]
                    [--left_illu_dorsal_det_data None]
                    [--right_illu_dorsal_det_data None]
                    [--save_separate_results "False"]
                    [--z_spacing None]
                    [--xy_spacing None]
```

## Installation

**Stable Release:** `pip install FUSE`<br>
**Development Head:** `pip install git+https://github.com/peng-lab/Leonardo-FUSE.git`


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


**MIT license**
