import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class FacialSkincareDataset(Dataset):
    def __init__(self, data_dir, csv_file):
        self.labels = pd.read_csv(csv_file)
        self.data_dir = data_dir

        self.data = self.get_data(self.data_dir)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ...

    def get_data(self, data_dir):
        files =  [os.path.join(data_dir, file) for file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, file))]