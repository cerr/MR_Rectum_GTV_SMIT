# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.transforms import OneOf
from monai.data import load_decathlon_datalist

from monai.transforms import MapTransform

class PercentileClipRescale(MapTransform):
    def __init__(self, keys, percentile=99.5, rescale_max=1000):
        super().__init__(keys)
        self.percentile = percentile
        self.rescale_max = rescale_max

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            # Assume image shape is (C, H, W, D) or (C, D, H, W) depending on data
            image_np = image.astype(np.float32)
            perc_val = np.percentile(image_np, self.percentile)
            image_np = np.clip(image_np, a_min=None, a_max=perc_val)
            image_np = (image_np / perc_val) * self.rescale_max
            d[key] = image_np.astype(np.int16)
        return d
    
def get_loader_v2(args):
    
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            PercentileClipRescale(keys=["image"], percentile=99.5, rescale_max=1000),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear", "nearest")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=["image","label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z)),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )