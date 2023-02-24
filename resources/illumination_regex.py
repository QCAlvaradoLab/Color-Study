import os
import glob

data_dir = "/home/hans/Desktop/Fish Illumination Correction/" # "/media/hans/T7/data/"
illumination_dirs = ["color_cluster_affecting/",
                "global_illumination/",
                "luderick/",
                "one_fish/",
                "suim/",
                "test_set_hard/", 
                "tree stump/"]

illumination_dirs = [os.path.join(data_dir, folder) for folder in illumination_dirs]

