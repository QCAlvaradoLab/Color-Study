import os
import glob

data_dir = "/home/hans/Downloads/data/" # "/media/hans/T7/data/"
folder_dirs = ["Cichlid Picture Collection REVISED (UPDATED)-20220403T172132Z-001/Cichlid Picture Collection REVISED (UPDATED)/*/",
               "deepfish/data/",
               "DeepFish/*/*/*/",
               #"f4k_detection_and_tracking/", #FLV
               "Fish_automated_identification_and_counting/luderick-seagrass/",
               "Fish-Pak/*/*/",
               "Fish Photography that needs to be matched for HK/*/",
               "foid/foid_images_v100/images/",
               "Machine learning training set/*/original image/",
               #"Cichlid Picture Collection REVISED (UPDATED)-20220403T172132Z-001/Cichlid Picture Collection REVISED (UPDATED)/Annotated Photos/",
               "pawsey/FDFML/frames/",
               "Phase 2 Color quantification for Hans/*/", 
               ]

roboflow_dirs = ["roboflow/Fish-43/",
                 "roboflow/Brackish-Underwater-2/",
                 "roboflow/Aquarium-Combined-3/",
               ]

roboflow = []
for dirname in roboflow_dirs:
    roboflow.extend([os.path.join(dirname, "train/"),
                     os.path.join(dirname, "valid/"),
                     os.path.join(dirname, "test/")])
folder_dirs.extend(roboflow)

folder_dirs = [os.path.join(data_dir, folder) for folder in folder_dirs]

