from folder_images_dataset import FolderImages

import os
import glob

import json
import cv2

from pprint import pprint 

import torch
from torch.utils.data import Dataset

INIT = ['whole_body']
HPARTS = [['ventral_side', 'anal_fin', 'pectoral_fin'], ['dorsal_side', 'dorsal_fin'], ['head', 'eye', 'operculum']]
INDEP = ['humeral_blotch', 'pelvic_fin', 'caudal_fin']

IMG_TYPES = ['jpg', 'png', 'arw']
IMG_TYPES.extend([x.upper() for x in IMG_TYPES])

#JOINT_TRANSFORMS = ['CenterCrop', 'FiveCrop', 'Pad', 'RandomAffine', 'RandomCrop', 'RandomHorizontalFlip', 
#                    'RandomVerticalFlip', 'RandomResizedCrop', 'RandomRotation', 'Resize', 'TenCrop']

#IMG_TRANSFORMS = ['ColorJitter', 'Grayscale', 'RandomGrayscale']
#IMG_FTRANSFORMS = ['adjust_gamma']

class FishDataset(Dataset):

    DATASET_TYPES = ["segmentation", "polygons"]
    DATASET_TYPES.extend(list(map(lambda s: \
                            s + "/composite", DATASET_TYPES)))

    def __init__(self, dataset_type="segmentation", config_file = "resources/config.json"):
        
        assert dataset_type in self.DATASET_TYPES
        
        with open(config_file, "r") as f:
            datasets_metadata = json.load(f)
        
        self.folder_path = datasets_metadata["folder_path"] 
        datasets = datasets_metadata["datasets"] 
        
        for data in datasets:
            
            dataset_getter = getattr(self, "get_%s_data" % data["name"])(data["type"], data["folder"]) 

    def get_coco_style_annotations(self, coco_images, coco_txt, ann_format="xywh"):

        assert ann_format in ["xywh", "xyxy"] and len(coco_images) == len(coco_txt)

        for image, objects_file in zip(coco_images, coco_txt):
            
            with open(objects_file, 'r') as f:
                obj = [x.strip() for x in f.readlines()]
                print(obj)
            num_objects = int(obj[0])
            h, w = [int(x) for x in obj[2].split(' ')]
            
            for idx in range(4, len(obj), 3):
                print (obj[idx])
                print (obj[idx+1])
                print (obj[idx+2])


            print (objects); exit()
            cv2.fillPoly(image, objects, 255 )
            cv2.imshow('f', image)
            

    def get_alvaradolab_data(self, dtype, path):
        
        assert dtype == "segmentation/composite"

        images = glob.glob(os.path.join(self.folder_path, path, '*.jpg'))
        labels = [x.replace(".jpg", ".txt") for x in images]

        return_value = self.get_coco_style_annotations(images, labels)

    def __len__(self):
        return 0

    def __getitem__(self, idx):         

        return torch.ones((5))

if __name__ == "__main__":

    dataset = FishDataset()

    
