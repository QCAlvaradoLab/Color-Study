import argparse

import os
import glob

import json
import cv2
import numpy as np

import torch

from torch.utils.data import Dataset, DataLoader

from . import display_composite_annotations
from . import colors, CPARTS, DATASET_TYPES
from . import dataset_splits


from .fish_coco_annotator import get_alvaradolab_data
from .fish_segmentation import get_ml_training_set_data

import traceback

#TODO ChainDataset

class FishDataset(Dataset):

    def __init__(self, dataset_type="segmentation", config_file = "resources/config.json", 
                    img_shape = 256, min_segment_positivity_ratio=0.0075, organs=["whole_body"],
                    dataset_split="train"): 
        # min_segment_positivity_ratio is around 0.009 - 0.011 for eye (the smallest part)
        
        global composite_labels

        assert dataset_type in DATASET_TYPES
        
        with open(config_file, "r") as f:
            datasets_metadata = json.load(f)
        
        self.folder_path = datasets_metadata["folder_path"] 
        datasets = datasets_metadata["datasets"] 
       
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.xy_pairs = []

        self.img_shape = img_shape
        
        # Accepts single type of data only
        datasets = reversed(list(reversed([x for x in datasets if x["type"] == dataset_type])))
        
        self.curated_images_count, self.dataset_generators = 0, []
        
        self.get_alvaradolab_data = get_alvaradolab_data
        self.get_ml_training_set_data = get_ml_training_set_data
        
        self.datasets, self.dataset_cumsum_lengths = [], []
        for data in datasets:
            
            dataset_method = "get_%s_data" % data["name"]
            
            try:
                dataset = getattr(self, dataset_method)(data["type"], data["folder"],
                                                        self.folder_path, 
                                                        img_shape, min_segment_positivity_ratio,
                                                        organs=organs) 
         
                # create train, val or test sets
                num_samples = {"train": [0, int(len(dataset) * dataset_splits["train"])]}
                num_samples["val"] = [num_samples["train"][1], num_samples["train"][1] + int(len(dataset) * dataset_splits["val"])] 
                num_samples["test"] = [num_samples["val"][1], num_samples["val"][1] + int(len(dataset) * dataset_splits["test"]) + 1]

                indices = range(*num_samples[dataset_split])
                dataset = torch.utils.data.Subset(dataset, indices)

                if len(self.dataset_cumsum_lengths) == 0:
                    self.dataset_cumsum_lengths.append(len(dataset))
                else:
                    self.dataset_cumsum_lengths.append(self.dataset_cumsum_lengths[-1] + len(dataset))

                self.datasets.append(dataset)
            
            except Exception as e:
                traceback.print_exc()
                print ("Write generator function for dataset: %s ;" % dataset_method, e)
        
        self.current_dataset_id = 0

    def __len__(self):
        return self.dataset_cumsum_lengths[-1]

    def __getitem__(self, idx):         
        
        while idx > self.dataset_cumsum_lengths[self.current_dataset_id]:
            self.current_dataset_id += 1

        dataset = self.datasets[self.current_dataset_id]
        
        if self.current_dataset_id == 0:
            prev_id = 0
        else:
            prev_id = self.current_dataset_id - 1

        data_index = idx - (self.dataset_cumsum_lengths[self.current_dataset_id] - self.dataset_cumsum_lengths[prev_id])
        
        return dataset[data_index]

if __name__ == "__main__":
   
    from . import composite_labels 
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize", default="alvaradolab", help="Flag to visualize composite labels")
    args = ap.parse_args()

    dataset = FishDataset(dataset_type="segmentation/composite", dataset_split="train") 
    print ("train", len(dataset))
    dataset = FishDataset(dataset_type="segmentation/composite", dataset_split="val") 
    print ("val", len(dataset))
    dataset = FishDataset(dataset_type="segmentation/composite", dataset_split="test") 
    print ("test", len(dataset))

    for data in dataset:
        image, segment = data
        display_composite_annotations(image, segment, composite_labels, dataset.min_segment_positivity_ratio)
