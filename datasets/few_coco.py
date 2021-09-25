# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from torchvision
# ------------------------------------------------------------------------

"""
Copy-Paste from torchvision, but add utility of caching images on memory
"""
from .coco import CocoDetection
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
import torch
from .builtin_meta import _get_coco_fewshot_instances_meta

from pathlib import Path

import torch
import torch.utils.data
from pycocotools import mask as coco_mask

from .torchvision_datasets import CocoDetection as TvCocoDetection
from ..util.misc import get_local_rank, get_local_size
from . import transforms as T
from .coco import make_coco_transforms, ConvertCocoPolysToMask


class BaseCocoDetection(TvCocoDetection):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, img_folder, ann_file, transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        super(BaseCocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        ret = _get_coco_fewshot_instances_meta()
        self.base_id_map = ret["base_dataset_id_to_contiguous_id"]

    def __getitem__(self, idx):
        img, target = super(BaseCocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        modified_target = []
        for anno in target:
            if anno["category_id"] in self.base_id_map:
                anno["category_id"] = self.base_id_map[anno["category_id"]]
                modified_target.append(anno)
        target = {'image_id': image_id, 'annotations': modified_target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

class FewShotCocoDetection(BaseCocoDetection):
    def __init__(self):
        pass
    


def build(image_set, args):
    root = Path(args.coco_path)
    dataset_name = args.dataset_name
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    if "base" in dataset_name:
        PATHS = {   
            "train": (root / "trainval2014", root / "annotations" / "trainvalno5k.json"),
            "val": (root / "val2014", root / "annotations" / "5k.json"),
        }

        img_folder, ann_file = PATHS[image_set]
        dataset = BaseCocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks,
                                cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size())

    return dataset

if __name__ == '__main__':
    root = "/mnt/disk1/huyvinuni/datasets/coco/"
    dataset_name = "base"
    mode = 'instances'
    if "base" in dataset_name:
        PATHS = {   
            "train": (root + "trainval2014", root + "annotations/" + "trainvalno5k.json"),
            "val": (root + "val2014", root + "annotations/" + "5k.json"),
        }

        img_folder, ann_file = PATHS["train"]
        dataset = BaseCocoDetection(img_folder, ann_file, transforms=make_coco_transforms("train"), return_masks=False,
                                cache_mode=False, local_rank=get_local_rank(), local_size=get_local_size())
        for img, target in dataset:
            print(target)
            break