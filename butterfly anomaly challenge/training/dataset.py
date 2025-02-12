from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd

class ButterflyDataset(Dataset):
    def __init__(self, data, root_dir, transforms=None):
        self.data = data
        self.root_dir = root_dir
        self.transforms = transforms

        # Validate the 'hybrid_stat' column to ensure it contains only expected values
        valid_classes = {"hybrid", "non-hybrid"}
        self.data["hybrid_stat"] = self.data["hybrid_stat"].str.strip().str.lower()  # Normalize the values
        if not set(self.data["hybrid_stat"].unique()).issubset(valid_classes):
            raise ValueError("Unexpected values found in 'hybrid_stat' column.")

        # Define classes explicitly to avoid relying on sorted order
        self.classes = ["non-hybrid", "hybrid"]
        self.cls_lbl_map = {cls: i for i, cls in enumerate(self.classes)}

        # Generate labels using a vectorized approach for efficiency
        self.labels = self.data["hybrid_stat"].map(self.cls_lbl_map).tolist()

        print("Created base dataset with {} samples".format(len(self.data)))
        
    def get_file_path(self, x):
        return os.path.join(self.root_dir, x['filename'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data.iloc[index]
        img_path = self.get_file_path(x)
        try:
            img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB format
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")
        
        lbl = self.labels[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, lbl




class ButterflyDatasetClass14(Dataset):
    def __init__(self, data: pd.DataFrame, root_dir: str, transforms=None):
        """
        Args:
            data (pd.DataFrame): DataFrame containing the CSV information.
            root_dir (str): Directory where the images are stored.
            transforms (callable, optional): Optional transform to be applied on an image.
        """
        self.data = data.copy()
        self.root_dir = root_dir
        self.transforms = transforms

        if "classification" not in self.data.columns:
            raise ValueError("CSV file must contain a 'classification' column.")
        
        # Parse classification and map multi-class cases to 14th class
        self.labels = self.data["classification"].apply(self.map_classification).tolist()

        print("Created dataset with {} samples".format(len(self.data)))

    @staticmethod
    def map_classification(classification: str):
        """
        Map the 'classification' string to a single numeric label (0-13 for single classes,
        14 for multi-class combinations like '8.0 and 9.0').
        
        Args:
            classification (str): The classification string to parse.
        
        Returns:
            int: Numeric class label (0-14).
        """
        try:
            if not isinstance(classification, str):
                classification = str(classification)            
            # Check if the classification is a multi-class combination
            if "and" in classification:
                return 14  # Map all multi-class cases to class 14
            else:
                # Convert single-class string to float and then to int
                return int(float(classification.strip()))
        except Exception as e:
            raise ValueError(f"Error parsing classification '{classification}': {e}")
        
    def get_file_path(self, x: dict) -> str:
        return os.path.join(self.root_dir, x['filename'])
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        img_path = self.get_file_path(row)
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Error loading image at {img_path}: {e}")
        
        # Retrieve the label
        lbl = self.labels[index]
        
        if self.transforms:
            img = self.transforms(img)
            
        return img, lbl
