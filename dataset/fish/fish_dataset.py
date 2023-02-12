import argparse

import os
import glob

import json
import cv2
import numpy as np

from pprint import pprint 

import torch

from torch.utils.data import IterableDataset, DataLoader

from . import display_composite_annotations
from . import colors, CPARTS, DATASET_TYPES

from .fish_coco_annotator import get_alvaradolab_data

import traceback

class FishDataset(IterableDataset):

    def __init__(self, dataset_type="segmentation", config_file = "resources/config.json", 
                    img_shape = 256, min_segment_positivity_ratio=0.0075): 
        # min_segment_positivity_ratio is around 0.009 - 0.011 for eye (the smallest part)
        
        assert dataset_type in DATASET_TYPES
        
        with open(config_file, "r") as f:
            datasets_metadata = json.load(f)
        
        self.folder_path = datasets_metadata["folder_path"] 
        datasets = datasets_metadata["datasets"] 
       
        self.composite_labels = list()
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.xy_pairs = []

        self.img_shape = img_shape
        
        # Accepts single type of data only
        datasets = reversed(list(reversed([x for x in datasets if x["type"] == dataset_type])))
        
        self.curated_images_count, self.dataset_generators = 0, []
        
        self.get_alvaradolab_data = get_alvaradolab_data
   
        for data in datasets:
            
            dataset_method = "get_%s_data" % data["name"]
            
            try:
                dataset_getter, dataset_count, self.composite_labels = getattr(self, dataset_method)(data["type"], data["folder"],
                                                                                self.composite_labels, self.folder_path, 
                                                                                img_shape, min_segment_positivity_ratio) 
                self.curated_images_count += dataset_count
                self.dataset_generators.append(dataset_getter)
            except Exception as e:
                traceback.print_exc()
                print ("Write generator function for dataset: %s ;" % dataset_method, e)

    def get_ml_training_set_data(self, dtype, path):

        assert dtype == "segmentation/composite"
        
        dataset_dirs = [x for x in glob.glob(os.path.join(self.folder_path, path, "*")) \
                            if os.path.isdir(x)]

        data = {}
        for dirname in dataset_dirs:
            pass
        
        #exit()

        #fish_folders = 

    #def __len__(self):
    #    return self.curated_images_count 

    def __iter__(self):         
        return self.dataset_generators[0]

if __name__ == "__main__":
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--visualize", default="alvaradolab", help="Flag to visualize composite labels")
    args = ap.parse_args()

    dataset = FishDataset(dataset_type="segmentation/composite") #DataLoader(, 
    #                        num_workers=1, batch_size=1)

    for image, segment in dataset:
        display_composite_annotations(image, segment, dataset.composite_labels, dataset.min_segment_positivity_ratio)
