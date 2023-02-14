from torch.utils.data import Dataset

class InMemoryDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(data)

    def __getitem__(self, idx):
        return data[idx]["image"], data[idx]["segments"]

