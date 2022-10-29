from hsv_color_thresholding import HSVColorClassifier
from gmm_colors import GMMColors, kwargs
from folder_images_dataset import folder_datasets

import configparser
import numpy as np

# HSV distance types: disparate, cartesian, gaussian

class ColorDistributions(object):
    
    def __init__(self, hsv_classifier, gmm_model):
        
        self.hsv_classifier = hsv_classifier
        self.gmm_model = gmm_model
        
    def predict_colors(self, img, dist_type = "gaussian"):
        
        """
        inner product
        dh = min(abs(h1-h0), 360-abs(h1-h0)) / 180.0
        ds = abs(s1-s0)
        dv = abs(v1-v0) / 255.0
        """
        
        # self.hsv_classifier.get_color()

        # img = img.reshape((-1, 3))
        
        if dist_type == "gaussian":
            
            colors = self.gmm_model.predict_colors(img)   
            clusters = np.unique(colors, return_counts=True)
            
            for color_group in clusters[0]:
                groups = np.where(colors == color_group)
                group_colors = img[groups]
                
                group_labels = self.hsv_classifier.get_color(group_colors)
                 
                print (groups.shape, clusters[1])


        pass 

if __name__ == "__main__":
    
    hsv_classifier = HSVColorClassifier("./resources/palette.png", gui=False)
    gmm_colors = GMMColors(folder_datasets, **kwargs)
    color_dist = ColorDistributions(hsv_classifier, gmm_colors) 
    
    # color_dist.predict_dataset_colors()
    
    print (folder_datasets, [len(x) for x in folder_datasets])
    for dataset in folder_datasets:
        for data in dataset:
            
            print ('data', data.shape)
            color_dist.predict_colors(data)
