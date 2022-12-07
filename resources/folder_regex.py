import os
import glob

data_dir = "/home/hans/Downloads/data/" # "/media/hans/T7/data/"
folder_dirs = ["Machine learning training set/*/original image/",
               "deepfish/data/",
               "Cichlid Picture Collection REVISED (UPDATED)-20220403T172132Z-001/Cichlid Picture Collection REVISED (UPDATED)/Annotated Photos/",
               "Cichlid Picture Collection REVISED (UPDATED)-20220403T172132Z-001/Cichlid Picture Collection REVISED (UPDATED)/*/",
               "DeepFish/*/*/*/",
               "Fish-Pak/*/*/",
               "Fish Photography that needs to be matched for HK/*/",
               "pawsey/FDFML/frames/",
               "Phase 2 Color quantification for Hans/*/"]

"""
folder_dirs.extend([
        "Fish_automated_identification_and_counting/luderick-seagrass/",
        ""
    ])
"""

folder_dirs = [os.path.join(data_dir, folder) for folder in folder_dirs]

