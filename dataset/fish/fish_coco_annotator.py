import sys
import gc
# sys computes data structure sizes for memory overflow handling

import os
import glob

import cv2
import numpy as np

from . import InMemoryDataset

import tracemalloc

def get_coco_style_annotations(coco_images, coco_txt, composite_labels, img_shape, min_segment_positivity_ratio, ann_format="xywh"):

    assert ann_format in ["xywh", "xyxy"] and len(coco_images) == len(coco_txt)
    
    data = []
    for index, (image, objects_file) in enumerate(zip(coco_images, coco_txt)):
        
        with open(objects_file, 'r') as f:
            obj = [x.strip() for x in f.readlines()]
        
        num_objects = int(obj[0])
        h, w = [int(x) for x in obj[2].split(' ')]
        
        image = cv2.imread(image)
        
        segment_array = np.zeros((img_shape, img_shape, len(composite_labels))) 
        
        for idx in range(4, len(obj), 4):
            organ = obj[idx]
            organ_index = composite_labels.index(organ)
            
            area_of_poly = float(obj[idx+1])
            
            if area_of_poly == 0:
                segment_array[:, :, organ_index] = -1
                continue

            poly_indices = [int(float(x)) for x in obj[idx+2].split(' ')]
            polygon = [(poly_indices[i], poly_indices[i+1]) 
                            for i in range(0, len(poly_indices)-1, 2)]
            
            seg = segment_array[:, :, organ_index].astype(np.uint8) 

            size_ratios = np.array([img_shape / float(image.shape[1]), img_shape / float(image.shape[0])]) 

            cv2.fillPoly(seg, [np.array(polygon * size_ratios).astype(np.int32)], 255) 

            if seg.sum() < (min_segment_positivity_ratio * img_shape * img_shape):
                seg.fill(-1)
            segment_array[:, :, organ_index] = seg 
             
        image = cv2.resize(image, (img_shape, img_shape))

        data.append({"image": image.transpose((2,0,1)), "segments": segment_array.transpose((2,0,1))})
        
        #print("index: ", index, [x / 1e6 for x in tracemalloc.get_traced_memory()], "MB")
        #index:  1000 [5446.751448, 5451.705668] MB
        #index:  2000 [10887.057792, 10887.674912] MB
        #index:  2534 [13792.174833, 13792.456553] MB
        #Killed

    return data

def get_alvaradolab_data(dtype, path, composite_labels, folder_path, img_shape, min_segment_positivity_ratio):
    
    tracemalloc.start()
    assert dtype == "segmentation/composite"

    images = glob.glob(os.path.join(folder_path, path, '*.jpg'))
    labels = [x.replace(".jpg", ".txt") for x in images]

    removable_indices = []
    for idx, (img, label) in enumerate(zip(images, labels)):
        if not (os.path.exists(img) and os.path.exists(label)):
            removable_indices.append(idx)
    
    for idx in reversed(removable_indices):
        del images[idx]
        del labels[idx]
    
    dataset_count = len(images)
    
    # get organ labels
    for txt_file in labels:
        with open(txt_file, 'r') as f:
            obj = [x.strip() for x in f.readlines()]

        for idx in range(4, len(obj), 4):
            organ = obj[idx]

            if not organ in composite_labels:
                composite_labels.append(organ) 

    print ("Using %d labeled images!" % len(images))
    data = get_coco_style_annotations(images, labels, composite_labels, img_shape, min_segment_positivity_ratio)
    
    dataset = InMemoryDataset(data)
    
    tracemalloc.stop()

    return dataset, dataset_count, composite_labels

