from folder_images_dataset import FolderImages

import os
import glob

import json
import cv2
import numpy as np

from pprint import pprint 

import torch
from torch.utils.data import Dataset

from color_constants import colors
from random import shuffle
colors = {k: colors[k] for k in colors if \
                any([x in colors for x in ["blue", "red", "cyan", "yellow", "green"]]) and \
                not any([str(x) in colors for x in range(1,5)])}
colors = list(colors.values())
shuffle(colors)

INIT = ['whole_body']

# Composite parts: 
# ventral_side seems like it needs to fully cover anal_fin and pectoral_fin from Google Search results on the topic
# dorsal_side doesn't cover dorsal fin
# operculum boundaries are outside head region
CPARTS = [['ventral_side', 'anal_fin', 'pectoral_fin'], ['dorsal_side', 'dorsal_fin'], ['head', 'eye', 'operculum']]

# Independent parts are ones without compositional overlap: whole_body contains these parts independently
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

    def __init__(self, dataset_type="segmentation", config_file = "resources/config.json", img_shape = 256):
        
        assert dataset_type in self.DATASET_TYPES
        
        with open(config_file, "r") as f:
            datasets_metadata = json.load(f)
        
        self.folder_path = datasets_metadata["folder_path"] 
        datasets = datasets_metadata["datasets"] 
       
        self.composite_labels = set()
        self.xy_pairs = []

        self.img_shape = img_shape
        
        # Accepts single type of data only
        datasets = reversed(list(reversed([x for x in datasets if x["type"] == dataset_type])))
        
        for data in datasets:
            
            dataset_getter = getattr(self, "get_%s_data" % data["name"])(data["type"], data["folder"]) 
            
            while True:
                try:
                    image, segment = next(dataset_getter)
                    self.display_composite_annotations(image, segment)
                                    
                except StopIteration:
                    break

            exit()
            #print (next(dataset_getter))

    def get_coco_style_annotations(self, coco_images, coco_txt, ann_format="xywh"):

        assert ann_format in ["xywh", "xyxy"] and len(coco_images) == len(coco_txt)

        for image, objects_file in zip(coco_images, coco_txt):
            
            with open(objects_file, 'r') as f:
                obj = [x.strip() for x in f.readlines()]
            
            num_objects = int(obj[0])
            h, w = [int(x) for x in obj[2].split(' ')]
            
            image = cv2.imread(image)
            
            segment_array = np.zeros((*image.shape[:2], len(self.composite_labels)))
            empty_indices = list(range(len(segment_array)))
            
            for idx in range(4, len(obj), 4):
                organ = obj[idx]
                area_of_poly = float(obj[idx+1])
                poly_indices = [int(float(x)) for x in obj[idx+2].split(' ')]
                polygon = [(poly_indices[i], poly_indices[i+1]) 
                                for i in range(0, len(poly_indices)-1, 2)]
                
                organ_index = self.composite_labels.index(organ)
                seg = segment_array[:, :, organ_index].astype(np.uint8) 
                cv2.fillPoly(seg, [np.array(polygon).astype(np.int32)], 255) 
                segment_array[:, :, organ_index] = seg
                
            yield image.transpose((2,0,1)), segment_array.transpose((2,0,1))

    def display_composite_annotations(self, image, labels_map, hide_whole_body_segment=True, show_composite_parts=True):
        
        alpha = 0.8

        image = image.transpose((1,2,0)).astype(np.uint8)
        cv2.imshow("image", image)
        
        if hide_whole_body_segment:
            largest_segment_id = np.argmax(labels_map.sum(axis=(1,2)))
            print ("Ignoring largest segment %s!" % self.composite_labels[largest_segment_id])
        else:
            largest_segment_id = -1

        labels_map = labels_map.transpose((1,2,0)).astype(np.uint8)

        outer_loop_times = len(CPARTS) if show_composite_parts and any([x in self.composite_labels for y in CPARTS for x in y]) else 1
        
        image_copy = image.copy()

        for outer_loop_idx in range(outer_loop_times):
            
            visited = []
            
            for seg_id in range(labels_map.shape[-1]):
                
                if outer_loop_times > 1:
                    
                    if self.composite_labels[seg_id] not in CPARTS[outer_loop_idx]:
                        continue
                    else:
                        visited.append(CPARTS[outer_loop_idx].index(self.composite_labels[seg_id]))

                cv2.imshow("fish_%s"%self.composite_labels[seg_id], labels_map[:,:,seg_id])
                
                if largest_segment_id != -1 and seg_id == largest_segment_id:
                    continue

                seg_image = np.expand_dims(labels_map[:,:,seg_id], axis=-1).repeat(3, axis=-1) * np.array(colors[seg_id]).astype(np.uint8)
                seg_image = cv2.addWeighted(image, 1, seg_image, 1, 1.0)
                image = cv2.addWeighted(image, 1-alpha, seg_image, alpha, 1.0)
            
            missing_annotation_indices = set(range(len(CPARTS[outer_loop_idx]))) - set(visited)
            if len(missing_annotation_indices) > 0:
                print ("Cannot find annotations for %s" % ", ".join([CPARTS[outer_loop_idx][x] for x in missing_annotation_indices])) 

            cv2.imshow("fish_%s"%( "all_parts" if outer_loop_times == 1 else ", ".join(CPARTS[outer_loop_idx])
                            ), image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
            image = image_copy

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
        
        return return_value_generator
    
    def get_ml_training_set_data(self, dtype, path):

        assert dtype == "segmentation/composite"

    def __len__(self):
        return 

    def __getitem__(self, idx):         

        return torch.ones((5))

if __name__ == "__main__":

    dataset = FishDataset(dataset_type="segmentation/composite")
