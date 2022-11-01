import glob
import os
from folder_images_dataset import FolderImages

data_dir = "/home/hans/Downloads/data/" # "/home/hans/Documents/data/"
FOLDERS = ["Machine learning training set/*/*/", # original image/",
           "deepfish/zip/*/",
           "Cichlid Picture Collection REVISED (UPDATED)-20220403T172132Z-001/Cichlid Picture Collection REVISED (UPDATED)/*/",
           "DeepFish/*/*/*/",
           "Fish-Pak/*/*/",
           "Fish Photography that needs to be matched for HK/*/",
           "pawsey/FDFML/frames/",
           "Phase 2 Color quantification for Hans/*/",
           "Fish_automated_identification_and_counting/luderick-seagrass/",
           "Fish-Pak/*/*",
           "roboflow/Aquarium-Combined-3/*/images/*/",
           "roboflow/Brackish-Underwater-2/*/images/*/",
           "roboflow/Fish-43/*/images/*/",
           "SUIM/SUIM/train_val/images/",
           "SUIM/SUIM/TEST/images/",
           "foid/foid_images_*/images/"
           ]
FOLDERS = [os.path.join(data_dir, x) for x in FOLDERS]

folder_datasets = []

for directory in FOLDERS:
    for dir_name in glob.glob(directory):
        if os.path.isdir(dir_name):
            # Remove all segmentation masks
            if "/masks/" not in dir_name:
                obj = FolderImages(dir_name, data_dir=data_dir)
                if len(obj)>0:
                    folder_datasets.append(obj)

print ("\n".join([str(x) for x in folder_datasets]))
print ("TOTAL: %d images!" % sum([len(x) for x in folder_datasets]))
