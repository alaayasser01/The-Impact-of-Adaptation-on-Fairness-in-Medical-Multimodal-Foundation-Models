import torch
import numpy as np
from PIL import Image
import os
from .base import BaseDataset

class DDI(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "fitzpatrick" or self.sens_name == "skin_tone":
            # Explicitly map groups to unique integers to maintain analytical consistency:
            # Group 0: Light Skin (FST I-II)
            # Group 1: Intermediate Skin (FST III-IV)
            # Group 2: Dark Skin (FST V-VI)
            if "fitzpatrick" in self.dataframe.columns:
                mapping = {"I-II": 0, "III-IV": 1, "V-VI": 2}
                sa_array = self.dataframe["fitzpatrick"].map(mapping).values.astype(np.float32)
            else:
                mapping = {12: 0, 34: 1, 56: 2}
                sa_array = self.dataframe["skin_tone"].map(mapping).values.astype(np.float32)
            
            # Safety fallback check if any anomalies sneak into string formatting
            if np.isnan(sa_array).any():
                sa_array = self.dataframe["fitzpatrick"].astype('category').cat.codes.values.astype(np.float32)
                
            return sa_array
            
        elif self.sens_name == "All":
            # Dummy array to bypass sensitive constraints if evaluating unstratified performance
            return np.zeros(len(self.dataframe)).astype(np.float32)
            
        else:
            raise NotImplementedError(f"Sensitive attribute '{self.sens_name}' is not recognized for DDI.")

    def set_label(self):
        # Maps target classes: malignant -> 1, benign -> 0
        Y = np.where(
            self.dataframe["label_str"] == "malignant",
            1,
            0,
        )
        self.class_nums = 2
        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        # Targets the images folder set up in your DDI.json configuration file
        img = Image.open(os.path.join(self.path_to_images, f"{item['path']}")).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img