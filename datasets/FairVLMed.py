import os
import numpy as np
from PIL import Image
import torch

from .base import BaseDataset


class FairVLMed10k(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None):
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # Female: 1, Male: 0
            sa_array = np.asarray(self.dataframe["gender"].str.lower() != "male").astype(np.float32)
            class_0_count = np.sum(sa_array == 0)
            class_1_count = np.sum(sa_array == 1)
            total_count_sa = len(sa_array)
            weight_class_0 = total_count_sa / (2 * class_0_count)
            weight_class_1 = total_count_sa / (2 * class_1_count)
            self.class_weights_sa = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
            return sa_array

        elif self.sens_name == "Age":
            # Senior (>60): 1, Non-Senior (<=60): 0
            return np.asarray(self.dataframe["age"] > 60).astype(np.float32)

        elif self.sens_name == "Race":
            # White: 1, Non-White: 0
            return np.asarray(self.dataframe["race"].str.lower() == "white").astype(np.float32)

        elif self.sens_name in ["Language", "language"]:
            # English: 1, Non-English: 0
            return np.asarray(self.dataframe["language"].str.lower() == "english").astype(np.float32)

        elif self.sens_name == "Ethnicity":
            # Hispanic: 1, Non-Hispanic/Unknown: 0
            return np.asarray(self.dataframe["ethnicity"].str.lower() != "non-hispanic").astype(np.float32)

        elif self.sens_name == "MaritalStatus":
            # Married/Partnered: 1, Unmarried (Single, Widowed, etc.): 0
            return np.asarray(self.dataframe["maritalstatus"].str.lower() == "married or partnered").astype(np.float32)

        elif self.sens_name == "All":
            return np.zeros(len(self.dataframe)).astype(np.float32)

        else:
            raise NotImplementedError(f"Sensitive attribute '{self.sens_name}' is not supported.")

    def set_label(self):
        Y = np.where(
            self.dataframe["glaucoma"] == "yes",
            1,
            0,
        )
        self.class_nums = 2
        return Y

    def get_img(self, idx):
        item = self.dataframe.iloc[idx]

        if self.path_to_labels is not None:
            img = Image.fromarray(self.tol_images[idx])
        else:
            img = Image.open(os.path.join(self.path_to_images, f"{item['path']}")).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img