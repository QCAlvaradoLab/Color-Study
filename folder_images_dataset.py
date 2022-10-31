import glob
import os 

import cv2
import rawpy

from torch import nn 
from torch.utils.data import Dataset

import traceback

class FolderImages(Dataset):
    
    def __init__(self, folder, img_size = 512):
        
        self.img_size = img_size
        self.images = glob.glob(os.path.join(folder, "*"))
        
        # Remove annotation files in txt if any!
        self.images = [x for x in self.images if not x.endswith(".txt")]

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
           
        try:
            if 'arw' in self.images[idx].lower():
                raw = rawpy.imread(self.images[idx])
                img = raw.postprocess()
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img = cv2.imread(self.images[idx])
            
            img = cv2.resize(img, (self.img_size, self.img_size))
            
        except Exception:
            traceback.print_exc()
            print ("Error with image %d/%d: " % (idx, len(self.images)), self.images[idx])
            if idx + 1 <= len(self.images) - 1:
                return self.__getitem__(idx+1)

        return img
   

data_dir = "/media/hans/T7/data/" # "/home/hans/Documents/data/"
folder_dirs = ["Machine learning training set/*/original image/",
               "deepfish/data/",
               "Cichlid Picture Collection REVISED (UPDATED)-20220403T172132Z-001/Cichlid Picture Collection REVISED (UPDATED)/Annotated Photos/",
               "Cichlid Picture Collection REVISED (UPDATED)-20220403T172132Z-001/Cichlid Picture Collection REVISED (UPDATED)/*/",
               "DeepFish/*/*/*/",
               "Fish-Pak/*/*/",
               "Fish Photography that needs to be matched for HK/*/",
               "Pawsey/FDFML/labelled/frames/*/",
               "Phase 2 Color quantification for Hans/*/"]

folder_dirs = [os.path.join(data_dir, folder) for folder in folder_dirs]

folder_datasets = []

for directory in folder_dirs:
    for dirs in glob.glob(directory):
        if os.path.isdir(dirs):
            folder_datasets.append(FolderImages(dirs))

# print ([len(x) for x in folder_datasets])
