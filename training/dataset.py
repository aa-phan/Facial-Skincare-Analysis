import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from matplotlib import pyplot as plt


# TODO: data augmentation: resize images to 256x256 ✅
# TODO: data augmentation: randomly rotate images 0 to 15 degree
# TODO: data augmentation: randomly alter brightness, contrast, saturation and hue
# TODO: data augmentation: randomly flip images horizontally or vertically

def map_class_labels(csv_file: str = "labels_fitzpatrick17k.csv") -> dict:
    df = pd.read_csv(csv_file)
    labels = df["label"].values
    return {label: i for i, label in enumerate(set(labels))}


class FacialSkincareDataset(Dataset):
    def __init__(self, data_dir: str, csv_file: str) -> None:
        """
        Args:
            data_dir (str): Path to the directory containing image files.
            csv_file (str): Path to the CSV file containing labels (and optionally file paths).
        """
        df = pd.read_csv(csv_file)
        self.labels = df["label"].values # NOTE: TOTAL 114 CLASSES

        self.files = Path(data_dir).glob("*")  # all files in the directory
        self.files = [f for f in self.files if f.is_file()]

        if len(self.labels) != len(self.files):
            raise ValueError(
                f"Number of labels ({len(self.labels)}) does not match "
                f"number of images ({len(self.files)})"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
        """
        Args:
            idx (int): index

        Returns:
            (image, label): Tuple of a PIL.Image and a label (e.g. int or string).
        """

        label = self.labels[idx]
        image_path = self.files[idx]

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((256, 256))
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute(2, 0, 1)

            label = map_class_labels()[label]
            label = torch.tensor(label, dtype=torch.long)
            return img, label

if __name__ == '__main__':
    dataset = FacialSkincareDataset(
        data_dir="data/fitzpatrick17k_data",
        csv_file="labels_fitzpatrick17k.csv"
    )
    image, label = dataset[0]
    print("Label:", label)
    breakpoint()