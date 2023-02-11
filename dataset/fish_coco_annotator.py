from folder_images_dataset import FolderImages

import argparse

import os
import glob

import json
import cv2
import numpy as np

from pprint import pprint 

import torch

#from torch.utils.data import Dataset
from torch.utils.data import IterableDataset, DataLoader

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
CPARTS.append(INDEP)
CPARTS.insert(0, INIT)

IMG_TYPES = ['jpg', 'png', 'arw']
IMG_TYPES.extend([x.upper() for x in IMG_TYPES])

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

    def display_composite_annotations(self, image, labels_map, hide_whole_body_segment=False, show_composite_parts=True):
        
        alpha = 0.8

        image = image.transpose((1,2,0)).astype(np.uint8)
        #cv2.imshow("image", image)
        
        if hide_whole_body_segment:
            largest_segment_id = np.argmax(labels_map.sum(axis=(1,2)))

            if self.composite_labels[largest_segment_id] == "whole_body":
                print ("\nIgnoring largest segment %s!" % self.composite_labels[largest_segment_id])
            else:
                print ("\nCannot find whole body segment!")
                largest_segment_id = -1
        else:
            largest_segment_id = 0

        labels_map = labels_map.transpose((1,2,0)).astype(np.uint8)
        
        outer_loop_times = len(CPARTS) if show_composite_parts and any([x in self.composite_labels for y in CPARTS for x in y]) else 1
        
        image_copy = image.copy()

        visited = []
        for outer_loop_idx in range(outer_loop_times):
            
            visited_cparts = []
            
            for seg_id in range(labels_map.shape[-1]):
                
                if outer_loop_times > 1:
                     
                    try:
                        if subset_ratio_denominator == 1.0:
                            seg_mask_ratio = 1.0 if seg_mask_ratio==0 else seg_mask_ratio
                            subset_ratio_denominator = seg_mask_ratio   
                    except NameError:
                        subset_ratio_denominator = 1.0

                    if self.composite_labels[seg_id] not in CPARTS[outer_loop_idx]:
                        continue
                    else:
                        seg_mask_ratio = np.sum(labels_map[:,:,seg_id]) / (255.0 * np.prod(labels_map.shape[:2]))
                        seg_mask_ratio = seg_mask_ratio / subset_ratio_denominator
                
                        print ("%s mask ratio wrt image: %f" % (self.composite_labels[seg_id] + \
                                                        ("" if "whole_body" == self.composite_labels[seg_id] else (
                                                        " subset ratio wrt whole_body" if subset_ratio_denominator!=1.0 else "")), 
                                                        seg_mask_ratio))

                        if seg_mask_ratio > self.min_segment_positivity_ratio:
                            visited_cparts.append(CPARTS[outer_loop_idx].index(self.composite_labels[seg_id]))
                        else:
                            print ("\n%s is too small wrt positivity ratio!\n" % self.composite_labels[seg_id])
                            continue

                cv2.imshow("fish_%s"%self.composite_labels[seg_id], labels_map[:,:,seg_id])
                
                seg_image = np.expand_dims(labels_map[:,:,seg_id], axis=-1).repeat(3, axis=-1) * np.array(colors[seg_id]).astype(np.uint8)
                seg_image = cv2.addWeighted(image, 1, seg_image, 1, 1.0)
                image = cv2.addWeighted(image, 1-alpha, seg_image, alpha, 1.0)
            
            missing_annotation_indices = set(range(len(CPARTS[outer_loop_idx]))) - set(visited_cparts)
            if len(missing_annotation_indices) > 0:
                print ("Cannot find annotations for %s" % ", ".join([CPARTS[outer_loop_idx][x] for x in missing_annotation_indices])) 
            
                if len(missing_annotation_indices) == len(CPARTS[outer_loop_idx]):
                    continue

            visited.append(visited_cparts)

            cv2.imshow("fish_%s"%( "all_parts" if outer_loop_times == 1 else ", ".join(CPARTS[outer_loop_idx])
                            ), image)
            cv2.waitKey()
            
            image = image_copy

        cv2.destroyAllWindows()
    
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
        dataset.display_composite_annotations(image, segment)
