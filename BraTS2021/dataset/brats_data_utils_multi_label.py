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

from sklearn.model_selection import KFold

import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm 
from torch.utils.data import Dataset 

def resample_img(
    image: sitk.Image,
    out_spacing = (2.0, 2.0, 2.0),
    out_size = None,
    is_label: bool = False,
    pad_value = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    # get original spacing and size
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    # convert our z, y, x convention to SimpleITK's convention
    out_spacing = list(out_spacing)[::-1]

    if out_size is None:
        # calculate output size in voxels
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]

    # determine pad value
    if pad_value is None:
        pad_value = image.GetPixelIDValue()

    # set up resampler
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    # perform resampling
    image = resample.Execute(image)

    return image

class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d  = self.read_data(datalist[i])
                self.cache_data.append(d)

    def read_data(self, data_path):
        
        file_identifizer = data_path.split("/")[-1].split("_")[-1]
        image_paths = [
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_t1.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_flair.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_t2.nii.gz"),
            os.path.join(data_path, f"BraTS2021_{file_identifizer}_t1ce.nii.gz")
        ]
        seg_path = os.path.join(data_path, f"BraTS2021_{file_identifizer}_seg.nii.gz")

        image_data = [sitk.GetArrayFromImage(sitk.ReadImage(p)) for p in image_paths]
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        
        original_shape = seg_data.shape

        image_data = np.array(image_data).astype(np.float32)
        seg_data = np.expand_dims(np.array(seg_data).astype(np.int32), axis=0)
        return {
            "image": image_data,
            "label": seg_data,
            "data_path": data_path,
            "file_identifizer": file_identifizer,
            "original_shape": original_shape,
        } 

    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else :
            try:
                image = self.read_data(self.datalist[i])
            except Exception as e:
                print(f"error: {e}")
                next_i = (i + 1) % len(self.datalist)
                if next_i != i:
                    return self.__getitem__(next_i)
                else:
                    raise RuntimeError("cannot load any data")
        if self.transform is not None :
            image = self.transform(image)
        
        return image

    def __len__(self):
        return len(self.datalist)

def get_kfold_data(data_paths, n_splits, shuffle=False):
    X = np.arange(len(data_paths))
    kfold = KFold(n_splits=n_splits, shuffle=shuffle)
    return_res = []
    for a, b in kfold.split(X):
        fold_train = []
        fold_val = []
        for i in a:
            fold_train.append(data_paths[i])
        for j in b:
            fold_val.append(data_paths[j])
        return_res.append({"train_data": fold_train, "val_data": fold_val})

    return return_res

class Args:
    def __init__(self) -> None:
        self.workers=8
        self.fold=0
        self.batch_size=2

def get_loader_brats(data_dir, batch_size=1, fold=0, fast_dev_run=False):

    all_dirs = os.listdir(data_dir)
    all_paths = [os.path.join(data_dir, d) for d in all_dirs if d.startswith("BraTS2021")]
    all_paths.sort()
    # stop shuffle paths
    # import random
    # random.shuffle(all_paths)
    size = len(all_paths)
    if fast_dev_run:
        size = 10
    print(f"BraTS2021 data size is {size}.")
    train_size = int(0.7 * size)
    val_size = int(0.1 * size)
    train_files = all_paths[:train_size]
    val_files = all_paths[train_size:train_size + val_size]
    test_files = all_paths[train_size+val_size:size]
    print(f"train is {len(train_files)}, val is {len(val_files)}, test is {len(test_files)}")

    train_transform = transforms.Compose(
        [   
            transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[96, 96, 96],
                                        random_size=False),
            transforms.SpatialPadd(keys=["image", "label"], spatial_size=(96, 96, 96)),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"],),
        ]
    )

    val_transform = transforms.Compose(
        [   transforms.ConvertToMultiChannelBasedOnBratsClassesD(keys=["label"]),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label", "original_shape"]),
        ]
    )
        

    train_ds = PretrainDataset(train_files, transform=train_transform)

    val_ds = PretrainDataset(val_files, transform=val_transform)
    
    test_ds = PretrainDataset(test_files, transform=val_transform)

    loader = [train_ds, val_ds, test_ds]

    return loader

def save_brats_pred_seg(
    prediction_array: np.ndarray,
    data_path: str,
    file_identifier: str,
    model_name: str,
    original_shape,
    foreground_start_coord,
    foreground_end_coord,
):
    if not os.path.exists(data_path):
        print(f"data_path {data_path} does not exist!")
        return

    pred_seg_path = os.path.join(data_path, f"BraTS2021_{file_identifier}_{model_name}_pred_seg.nii.gz")
    
    label_map = np.zeros(prediction_array.shape[1:], dtype=np.uint8)
    if prediction_array.shape[0] == 3:
        label_map[prediction_array[1] == 1] = 2  # ED
        label_map[prediction_array[0] == 1] = 1  # NCR/NET
        label_map[prediction_array[2] == 1] = 4  # ET
    else:
        label_map = prediction_array.astype(np.uint8)

    # pad back to original shape
    original_label_map = np.zeros(original_shape, dtype=np.uint8)
    
    s_z, e_z = foreground_start_coord[0], foreground_end_coord[0]
    s_y, e_y = foreground_start_coord[1], foreground_end_coord[1]
    s_x, e_x = foreground_start_coord[2], foreground_end_coord[2]
    
    original_label_map[s_z:e_z, s_y:e_y, s_x:e_x] = label_map

    pred_image = sitk.GetImageFromArray(original_label_map)
    
    original_image_path = os.path.join(data_path, f"BraTS2021_{file_identifier}_t1.nii.gz")
    if os.path.exists(original_image_path):
        original_image = sitk.ReadImage(original_image_path)
        pred_image.SetOrigin(original_image.GetOrigin())
        pred_image.SetSpacing(original_image.GetSpacing())
        pred_image.SetDirection(original_image.GetDirection())
    else:
        print(f"original_image_path {original_image_path} does not exist!")
        return

    sitk.WriteImage(pred_image, pred_seg_path)

    print(f"save pred_seg to {pred_seg_path}")

def save_brats_uncer(
    uncer_array: np.ndarray,
    data_path: str,
    file_identifier: str,
    model_name: str,
    original_shape,
    foreground_start_coord,
    foreground_end_coord,
):
    if not os.path.exists(data_path):
        print(f"data_path {data_path} does not exist!")
        return

    uncer_save_path = os.path.join(data_path, f"BraTS2021_{file_identifier}_{model_name}_uncer.npz")
    
    # pad back to original shape
    # uncer_array shape is (10, 3, crop_z, crop_y, crop_x)
    original_uncer_map = np.zeros((uncer_array.shape[0], uncer_array.shape[1], *original_shape), dtype=np.uint8)
    
    s_z, e_z = foreground_start_coord[0], foreground_end_coord[0]
    s_y, e_y = foreground_start_coord[1], foreground_end_coord[1]
    s_x, e_x = foreground_start_coord[2], foreground_end_coord[2]
    
    # 将不确定性值从0-1范围转换为0-100整数范围
    uncer_array_scaled = (uncer_array * 100).astype(np.uint8)
    original_uncer_map[:, :, s_z:e_z, s_y:e_y, s_x:e_x] = uncer_array_scaled

    np.savez_compressed(uncer_save_path, uncer=original_uncer_map)

    print(f"save uncer to {uncer_save_path}")
