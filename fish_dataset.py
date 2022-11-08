from folder_images_dataset import FolderImages

from torch.utils.data import Dataset

class FishDataset(Dataset):

    DATASET_TYPES = ["segmentation", "segmentation/composite", "polygons", "polygons/composite"]

    def __init__(self, dataset_type=""):
        
        assert dataset
