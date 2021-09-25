import torch
from argparse import ArgumentParser
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import backbone
from datasets.data_prefetcher import data_prefetcher
from datasets.coco import make_coco_transforms
from datasets.few_coco import BaseCocoDetection
import torchvision
from torchvision.ops import roi_align
import pickle as pkl

def argument_parse():
    parser = ArgumentParser()
    parser.add_argument("--num-classes", default= 60, type=int, help="number of classes")
    parser.add_argument("--hidden-dim", default=256, type=int, help="hidden dimension of class queries")
    parser.add_argument("--feature-extractor", default="swav-resnet50", type=str, help="feature extractor")
    parser.add_argument("--dataset-name", default="base", type=str, help="dataset name")
    parser.add_argument("--root", default="/mnt/disk1/huyvinuni/datasets/coco/" , type=str, help="dataset root")
    parser.add_argument("--js", default="trainvalno5k.json", type=str, help="json file")
    return parser.parse_args()

def extract_roi(features, targets):
    # bs,c,h,w = features.size()
    targets = targets[0]
    srcs = {k: nested_tensors.decompose[0] for k, nested_tensors in features.items()}
    target_features = {k: {} for k in set(targets.labels.tolist())}
    for i in range(targets["boxes"].shape[0]):
        for k, feature_map in srcs.items():
            spatial_size = feature_map.size()[-2:]
            x1, y1, x2, y2 = targets["boxes"][i, :].tolist()
            roi_feat = roi_align(feature_map, [x1, y1, x2, y2], 1)
            roi_feat = roi_feat.view(-1)
            label = targets["labels"][i]
            target_features[label][k] = roi_feat

    return target_features

def main():
    args = argument_parse()
    num_classes = args.num_classes
    hidden_dim = args.hidden_dim
    feature_extractor = args.feature_extractor
    if args.dataset_name == "base":
        setz = "trainval2014"

    device = "cuda:0"

    dataset = BaseCocoDetection(os.path.join(args.root, setz), os.path.join(args.root, "annotations", args.js),
                                        transforms=make_coco_transforms("class_queries_gen"), return_masks=False, cache_mode=False, 
                                        local_rank=get_local_rank(), local_size=get_local_size())

    class_queries = {k: {layer: list(torch.tensor) for layer in ["0", "1", "2"]} for k in dataset_train.base_id_map.keys()}
    reverse_id_map = {v: k for k, v in dataset_train.base_id_map.items()}
    data_loader_train = DataLoader(dataset, batch_size=1, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)

    resnet50 = backbone.Backbone("resnet50", False, True, True)
    resnet50.eval()
    prefetcher = data_prefetcher(data_loader_train, device, prefetch=True)
    samples, targets = prefetcher.next()

    for _ in range(len(data_loader_train)):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features = resnet50(samples)
        target_features = extract_roi(features, targets)
        target_features = {reverse_id_map[k]: i for (k,i) in target_features}
        for cls in target_features.keys():
            class_queries[cls].extend(target_features[cls])
        samples, targets = prefetcher.next()
    with open("base_class_queries.pkl", "rb") as f:
        pkl.dump(class_queries, f)


    # generate class queries
    