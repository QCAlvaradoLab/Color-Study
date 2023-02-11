from folder_images_dataset import FolderImages

import argparse

import os
import glob

import json
import cv2
import numpy as np

from pprint import pprint 

import torch

from torch.utils.data import IterableDataset, DataLoader

from .visualize_composite_labels import display_composite_annotations
from . import colors, CPARTS
print (CPARTS)
#JOINT_TRANSFORMS = ['CenterCrop', 'FiveCrop', 'Pad', 'RandomAffine', 'RandomCrop', 'RandomHorizontalFlip', 
#                    'RandomVerticalFlip', 'RandomResizedCrop', 'RandomRotation', 'Resize', 'TenCrop']

#IMG_TRANSFORMS = ['ColorJitter', 'Grayscale', 'RandomGrayscale']
#IMG_FTRANSFORMS = ['adjust_gamma']

class FishDataset(IterableDataset):

    DATASET_TYPES = ["segmentation", "polygons"]
    DATASET_TYPES.extend(list(map(lambda s: \
                            s + "/composite", DATASET_TYPES)))

    def __init__(self, dataset_type="segmentation", config_file = "resources/config.json", img_shape = 256, min_segment_positivity_ratio=0.0075): 
        # min_segment_positivity_ratio is around 0.009 - 0.011 for eye (the smallest part)
        
        assert dataset_type in self.DATASET_TYPES
        
        with open(config_file, "r") as f:
            datasets_metadata = json.load(f)
        
        self.folder_path = datasets_metadata["folder_path"] 
        datasets = datasets_metadata["datasets"] 
       
        self.composite_labels = set()
        self.min_segment_positivity_ratio = min_segment_positivity_ratio
        self.xy_pairs = []

        self.img_shape = img_shape
        
        # Accepts single type of data only
        datasets = reversed(list(reversed([x for x in datasets if x["type"] == dataset_type])))
        
        self.curated_images_count, self.dataset_generators = 0, []
        
        for data in datasets:
            
            dataset_method = "get_%s_data" % data["name"]
            
            try:
                dataset_getter, dataset_count = getattr(self, dataset_method)(data["type"], data["folder"]) 
                self.curated_images_count += dataset_count
                self.dataset_generators.append(dataset_getter)
            except Exception:
                print ("Write generator function for dataset: %s" % dataset_method)

    def get_coco_style_annotations(self, coco_images, coco_txt, ann_format="xywh"):

        assert ann_format in ["xywh", "xyxy"] and len(coco_images) == len(coco_txt)

        for image, objects_file in zip(coco_images, coco_txt):
            
            with open(objects_file, 'r') as f:
                obj = [x.strip() for x in f.readlines()]
            
            num_objects = int(obj[0])
            h, w = [int(x) for x in obj[2].split(' ')]
            
            image = cv2.imread(image)
            
            segment_array = np.zeros((self.img_shape, self.img_shape, len(self.composite_labels))) 
            empty_indices = list(range(len(segment_array)))
            
            for idx in range(4, len(obj), 4):
                organ = obj[idx]
                area_of_poly = float(obj[idx+1])
                poly_indices = [int(float(x)) for x in obj[idx+2].split(' ')]
                polygon = [(poly_indices[i], poly_indices[i+1]) 
                                for i in range(0, len(poly_indices)-1, 2)]
                
                organ_index = self.composite_labels.index(organ)
                seg = segment_array[:, :, organ_index].astype(np.uint8) 

                size_ratios = np.array([self.img_shape / float(image.shape[1]), self.img_shape / float(image.shape[0])]) 

                cv2.fillPoly(seg, [np.array(polygon * size_ratios).astype(np.int32)], 255) 
                segment_array[:, :, organ_index] = seg 

                if (seg.sum() / 255.0) < (self.min_segment_positivity_ratio * self.img_shape * self.img_shape):
                    seg.fill(-1)
                 
            image = cv2.resize(image, (self.img_shape, self.img_shape))

            yield image.transpose((2,0,1)), segment_array.transpose((2,0,1))

   
    def get_segmentation_annotations(self, images, labels):
        
        pass

    def get_alvaradolab_data(self, dtype, path):
        
        assert dtype == "segmentation/composite"

        images = glob.glob(os.path.join(self.folder_path, path, '*.jpg'))
        labels = [x.replace(".jpg", ".txt") for x in images]

        removable_indices = []
        for idx, (img, label) in enumerate(zip(images, labels)):
            if not (os.path.exists(img) and os.path.exists(label)):
                removable_indices.append(idx)
        
        for idx in reversed(removable_indices):
            del images[idx]
            del labels[idx]
        
        dataset_count = len(images)
        
        self.composite_labels = set(self.composite_labels)
        
        # get organ labels
        for txt_file in labels:
            with open(txt_file, 'r') as f:
                obj = [x.strip() for x in f.readlines()]

            for idx in range(4, len(obj), 4):
                organ = obj[idx]
                self.composite_labels.add(organ) 
        
        self.composite_labels = list(self.composite_labels)

        print ("Using %d labeled images!" % len(images))
        return_value_generator = self.get_coco_style_annotations(images, labels)
        
        return return_value_generator, dataset_count
    
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
