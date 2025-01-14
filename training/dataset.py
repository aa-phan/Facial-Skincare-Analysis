import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

class DatasetProcessing(Dataset):
    def __init__(self):
        csv_path = "Classification/NNEW_trainval_0.txt"
        self.df = pd.read_csv(csv_path, sep='\\s+', header=None, names=['img', 'label', 'lesion'])
        self.df = self.df.dropna()

    def __getitem__(self, index):
        img, label, _ = self.df.iloc[index]

        img_path = os.path.join("Classification","JPEGImages", img)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        label = torch.tensor(label)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)

        return image, label

    def __len__(self):
        return len(self.df)


if __name__ == "__main__":
    ds = DatasetProcessing()
    print(ds[0])
    breakpoint()