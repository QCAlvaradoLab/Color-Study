import glob
import os 

import cv2
import rawpy

from torch import nn 
from torch.utils.data import Dataset

import traceback
import imageio

from pathlib import Path

class FolderImages(Dataset):
    
    def __init__(self, folder, data_dir = None, img_size = 512):
        
        self.img_size = img_size
        
        self.folder = folder if not data_dir else os.path.join(data_dir, folder)
        if self.folder[-1] == "/":
            self.folder = self.folder[:-1]
        
        self.data_dir = data_dir + "/"
        self.images = glob.glob(os.path.join(self.folder, "*"))
        
        # Remove annotation files in txt if any!
        self.images = [x for x in self.images if not x.endswith(".txt")]
    
    def get_color_stats(self):
        
        # HSV Color Ranges!
        color_ranges = [(181, -1), (256, -1), (256, -1)]

        #for img in self:
        #    for color in range(img.shape[-1]):
                #min_color, max_color = 
                #if color_ranges[color][0] > img.min(axis=2)

    def __len__(self):
        return len(self.images)
    
    def __str__(self):
        if self.data_dir:
            folder_name = self.folder.replace(self.data_dir, "")
        return "Folder: %s #Images: %d" % (folder_name, len(self.images))

    def __getitem__(self, idx):
        try:
            if 'arw' in self.images[idx].lower():
                raw = rawpy.imread(self.images[idx])
                img = raw.postprocess()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            elif 'gif' in self.images[idx].lower():
                gif = imageio.mimread(self.images[idx])
                print("Choosing frame 1/%d from the gif!" % len(gif))
                img = cv2.cvtColor(gif[0], cv2.COLOR_RGB2BGR)

            else:
                img = cv2.imread(self.images[idx])

            img = cv2.resize(img, (self.img_size, self.img_size))
            
        except Exception:
            traceback.print_exc()
            print ("Error with image %d/%d: " % (idx, len(self.images)), self.images[idx])
            if idx + 1 <= len(self.images) - 1:
                return self.__getitem__(idx+1)

        return img

from resources import folder_dirs

folder_datasets = []

for directory in folder_dirs:
    for dirs in glob.glob(directory):
        if os.path.isdir(dirs):
            # Remove all segmentation masks
            if "/masks/" not in dirs:
                if dirs[-1] == "/":
                    dirs = dirs[:-1]
                
                obj = FolderImages(dirs.replace(str(Path(dirs).parents[1])+"/", ""), str(Path(dirs).parents[1]))
                if len(obj)>0:
                    folder_datasets.append(obj)

