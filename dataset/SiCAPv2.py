import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List
import torchvision
from torchvision import transforms

class SiCAPv2(Dataset):
    """
    The SiCAPv2 class represents a Histopathology dataset of pixel-level annotations of prostate patches with different Gleason Grades.
     It loads images from a specified directory and their corresponding labels from a CSV file. The class supports optional
     transformations to preprocess the images and provides methods for accessing the dataset in a way suitable for training machine learning models.
    """
    def __init__(self, csv_file: str, root_dir: str, im_channels: int = 3):
        """
        Initialize the SiCAPv2 dataset.

        Args:
            csv_file (str): Path to the CSV file containing image names and labels.
            root_dir (str): Path to the directory where the images are stored.
            transform (torchvision.transforms.Compose): Transform to apply to the images.          

        Note:
            The transform must convert the PIL images to Tensor (in addition to the rest of trasnformations you want).
        """
        # Dictionary mapping class names to numerical indices
        self.dictionary = {'NC': 0, 'G3': 1, 'G4': 2, 'G5': 3, 'G4C': 2}

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * im_channels, std=[0.5] * im_channels)
        ])


        # Load the CSV file into a DataFrame
        dataframe = pd.read_excel(csv_file)

        # Transform labels into a format suitable for the model
        self.image_names, self.labels = self.transform_labels(dataframe, self.dictionary)

        self.root_dir = root_dir  # Store the directory of images
        self.image_list: List[torch.Tensor] = []  # List to store images
        self.labels_list: List[torch.Tensor] = []  # List to store labels

        print('Loading images to memory...')
        # Load images and their labels
        for position, element in enumerate(tqdm(self.image_names, desc="Loading")):
            img_path = os.path.join(self.root_dir, element)  # Create full image path
            try:
                image = Image.open(img_path).convert('RGB')  # Open image and convert to RGB
                y_label = torch.tensor(self.labels[position], dtype=torch.long)  # Convert label to tensor
                self.image_list.append(image)  # Store the image in the list
                self.labels_list.append(y_label)  # Store the label in the list
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")  # Handle image loading errors

    def change_transform(self, transform: torchvision.transforms.Compose) -> None:
        """
        Change the transformation applied to the images in the dataset.

        Args:
            transform (torchvision.transforms.Compose): New transformation to apply to the images.
        """
        self.transform = transform  # Update the transformation

    @staticmethod
    def transform_labels(dataframe: pd.DataFrame, dictionary_classes: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform labels in the DataFrame into a format that can be used by the model.

        Args:
            dataframe (pd.DataFrame): DataFrame containing image names and labels.
            dictionary_classes (dict): A dictionary mapping text labels to numerical indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the following elements:

                - **X (np.ndarray)**: A 0D tensor that represents the names of the image names in string format.
                - **y (np.ndarray)**: A 0D tensor that represents the transformed labels.
        
        """

        X, y = [], []  # Lists to store image names and labels

        for i, row in dataframe.iterrows():

            label = None  # Initialize label variable

            # Identify which label column has a 1
            for class_name, class_index in dictionary_classes.items():
                if row.get(class_name, 0) == 1:  # Check if the class column has a 1
                    label = class_index
                    break

            if label is None:
                continue
            X.append(row[0])  # Store image name
            y.append(label)

        return np.array(X), np.array(y)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return an element from the dataset given its index.

        Args:
            index (int): Index of the element to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the following elements:

                - **image (torch.Tensor)**: A 3D tensor of shape (num_chn, W, H) that represents the transformed image, where:
                    
                    - **num_chn**: Number of input channels (e.g., 3 for RGB images).
                    - **W**: The width of the image in pixels.
                    - **H**: The height of the image in pixels.
                - **label (torch.Tensor)**: A 0D tensor that represents the corresponding label.
        """
        image = self.image_list[index]  # Retrieve the image
        label = self.labels_list[index]  # Retrieve the corresponding label

        if self.transform:
            image = self.transform(image)  # Apply the transformation to the image if available

        return {
            "data": image,
            "cond": label,
        }


