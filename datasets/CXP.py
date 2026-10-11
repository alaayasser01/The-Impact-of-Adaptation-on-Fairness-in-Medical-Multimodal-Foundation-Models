import os
import numpy as np
from PIL import Image
import torch

from datasets.base import BaseDataset


class CXP(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # female: 1, male: 0
            sa_array = np.asarray(self.dataframe["Sex"].values != "Male").astype(np.float32)
            class_0_count = np.sum(sa_array == 0)
            class_1_count = np.sum(sa_array == 1)
            total_count_sa = len(sa_array)
            weight_class_0 = total_count_sa / (2 * class_0_count)
            weight_class_1 = total_count_sa / (2 * class_1_count)
            self.class_weights_sa = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
            return sa_array

        elif self.sens_name == "Age":
            # 4 Quartiles: Q1 (<49), Q2 (49-61), Q3 (62-73), Q4 (74+)
            ages = self.dataframe["Age"].values.astype(np.float32)
            sens_array = np.zeros_like(ages)
            sens_array[(ages >= 49) & (ages < 62)] = 1.0
            sens_array[(ages >= 62) & (ages < 74)] = 2.0
            sens_array[ages >= 74] = 3.0
            return sens_array

        elif self.sens_name == "All":
            return np.zeros(len(self.dataframe)).astype(np.float32)

        else:
            raise NotImplementedError(f"Sensitive attribute '{self.sens_name}' is not supported for CheXpert.")

    def set_label(self):
        Y = np.asarray(self.dataframe["No Finding"].values).astype(np.float32)
        self.class_nums = 2
        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]
        if self.path_to_labels is not None:
            img = Image.fromarray(self.tol_images[idx])
        else:
            img = Image.open(os.path.join(self.path_to_images, f"{item['Path']}")).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img