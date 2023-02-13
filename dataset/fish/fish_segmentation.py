import os
import glob

import rawpy
import cv2
import numpy as np

def imread(file_path):
    
    if ".arw" not in file_path.lower():
        return cv2.imread(file_path)
    else:
        img = rawpy.imread(file_path) 
        img = img.postprocess() 
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

def get_ml_training_set_annotations(segmentation_data, composite_labels, img_shape, min_segment_positivity_ratio):

    for data_key in segmentation_data.keys():
        
        image_path, segments_paths = segmentation_data[data_key]["image"], segmentation_data[data_key]["segments"]
        image = imread(image_path)
        
        segment_array = np.zeros((img_shape, img_shape, len(composite_labels))) 
        
        for organ_index, organ in enumerate(composite_labels):
            
            try:
                segment = imread(segments_paths[organ])
            
                segment = cv2.resize(segment, (img_shape, img_shape))
                segment = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
                segment[segment > 245] = 0
                segment[segment != 0] = 255
                
                area_of_segment = segment.sum() / 255.0
                
                if area_of_segment * 255 < (min_segment_positivity_ratio * img_shape * img_shape):
                    segment.fill(-1)
                
                segment_array[:, :, organ_index] = segment 
            
            except Exception:
                segment_array[:, :, organ_index].fill(-1) 
        
        image = cv2.resize(image, (img_shape, img_shape))

        yield image.transpose((2,0,1)), segment_array.transpose((2,0,1))

def get_ml_training_set_data(dtype, path, composite_labels, folder_path, img_shape, min_segment_positivity_ratio):
    
    #TODO: 9 missing images from dataset!

    assert dtype == "segmentation/composite"
    
    folders = [x for x in glob.glob(os.path.join(folder_path, path, "*")) \
                if os.path.isdir(x)]
    
    data = {}
    for directory in folders:
        
        dir_folders = glob.glob(os.path.join(directory, "*"))
        
        images = glob.glob(os.path.join(directory, 'original image/*'))
        for image_path in images:
            fname = "/".join(image_path.split('/')[-3:])
            search_key = '.'.join(fname.split('/')[-1].split('.')[:-1])
            data_index = os.path.join(directory.split('/')[-1], search_key)
            
            segments_path = glob.glob(os.path.join(directory, "*", search_key + "*"))
            organs = [x.split('/')[-2] for x in segments_path]
            organs.remove("original image")
            
            if not os.path.exists(image_path):
                #TODO print (image_path)
                continue

            segment_paths = {}
            for organ in organs:
                ann_paths = glob.glob(os.path.join(directory, organ, search_key + "*")) 
                
                organ = organ.replace(" ", "_")
                if not organ in composite_labels:
                    composite_labels.append(organ)
                if len(ann_paths) == 1:
                    if os.path.exists(ann_paths[0]):
                        segment_paths[organ] = ann_paths[0]  
            
            if len(segment_paths) > 0:
                data[data_index] = {"image": image_path, \
                                    "segments": segment_paths}
            #TODO 
            '''
            else:
                print (image_path, segment_paths)
        print (len(data))
            '''

    dataset_count = len(data)
    
    print ("Using %d labeled images!" % dataset_count)
    return_value_generator = get_ml_training_set_annotations(data, composite_labels, img_shape, min_segment_positivity_ratio)
    
    return return_value_generator, dataset_count, composite_labels

