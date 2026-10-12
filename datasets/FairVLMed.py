import os
import numpy as np
from PIL import Image
import torch

from .base import BaseDataset


class FairVLMed10k(BaseDataset):
    def __init__(self, dataframe, sens_name, transform, path_to_images=None, path_to_pickles=None, binarize=False):
        self.binarize = binarize
        super().__init__(dataframe, sens_name, transform, path_to_images, path_to_pickles)

    def set_sensitive(self):
        if self.sens_name == "Sex":
            # Female: 1, Male: 0
            sa_array = np.asarray(self.dataframe["gender"].astype(str).str.lower() != "male").astype(np.float32)
            class_0_count = np.sum(sa_array == 0)
            class_1_count = np.sum(sa_array == 1)
            total_count_sa = len(sa_array)
            weight_class_0 = total_count_sa / (2 * class_0_count) if class_0_count > 0 else 1.0
            weight_class_1 = total_count_sa / (2 * class_1_count) if class_1_count > 0 else 1.0
            self.class_weights_sa = torch.tensor([weight_class_0, weight_class_1], dtype=torch.float32)
            return sa_array

        elif self.sens_name == "Age":
            if getattr(self, "binarize", False):
                # Binarized: Senior (>60): 1, Non-Senior (<=60): 0
                return np.asarray(self.dataframe["age"] > 60).astype(np.float32)
            # Native Cardinality: 4 Age Quartiles (matching paper & thesis)
            if "age_group" in self.dataframe.columns:
                return self.dataframe["age_group"].values.astype(np.float32)
            elif "age" in self.dataframe.columns:
                return np.asarray(pd.qcut(self.dataframe["age"], q=4, labels=False, duplicates="drop")).astype(np.float32)
            return np.zeros(len(self.dataframe)).astype(np.float32)

        elif self.sens_name == "Race":
            if getattr(self, "binarize", False):
                # Binarized: White: 1, Non-White: 0
                return np.asarray(self.dataframe["race"].astype(str).str.lower() == "white").astype(np.float32)
            # Native Cardinality: Preserves all discrete race categories
            if "race_group" in self.dataframe.columns:
                return self.dataframe["race_group"].values.astype(np.float32)
            return self.dataframe["race"].astype("category").cat.codes.values.astype(np.float32)

        elif self.sens_name in ["Language", "language"]:
            if getattr(self, "binarize", False):
                # Binarized: English: 1, Non-English: 0
                return np.asarray(self.dataframe["language"].astype(str).str.lower() == "english").astype(np.float32)
            # Native Cardinality: Preserves discrete language groups
            if "language_group" in self.dataframe.columns:
                return self.dataframe["language_group"].values.astype(np.float32)
            return self.dataframe["language"].astype("category").cat.codes.values.astype(np.float32)

        elif self.sens_name == "Ethnicity":
            if getattr(self, "binarize", False):
                # Binarized: Hispanic: 1, Non-Hispanic/Unknown: 0
                return np.asarray(self.dataframe["ethnicity"].astype(str).str.lower() != "non-hispanic").astype(np.float32)
            # Native Cardinality
            return self.dataframe["ethnicity"].astype("category").cat.codes.values.astype(np.float32)

        elif self.sens_name == "MaritalStatus":
            if getattr(self, "binarize", False):
                # Binarized: Married: 1, Unmarried: 0
                return np.asarray(self.dataframe["maritalstatus"].astype(str).str.lower() == "married or partnered").astype(np.float32)
            # Native Cardinality: Preserves all granular marital categories
            return self.dataframe["maritalstatus"].astype("category").cat.codes.values.astype(np.float32)

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
            raw_path = str(item["path"])
            candidates = [
                os.path.join(self.path_to_images, raw_path),
                raw_path,
            ]
            if self.path_to_images:
                candidates.append(os.path.join(self.path_to_images, "Dataset", raw_path))
                candidates.append(os.path.join(self.path_to_images, raw_path.replace("Dataset/", "")))
                parent = os.path.dirname(self.path_to_images.rstrip("/\\"))
                candidates.append(os.path.join(parent, raw_path))
                candidates.append(os.path.join(parent, "Dataset", raw_path))

            full_path = None
            for p in candidates:
                if os.path.exists(p):
                    full_path = p
                    break
            if full_path is None:
                full_path = candidates[0]

            img = Image.open(full_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img