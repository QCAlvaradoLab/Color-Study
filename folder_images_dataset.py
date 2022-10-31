import glob
import os 

import cv2
import rawpy

from torch import nn 
from torch.utils.data import Dataset

import traceback
import imageio

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
   


