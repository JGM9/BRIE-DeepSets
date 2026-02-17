import pickle
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from pytorch_lightning import LightningDataModule
import pandas as pd
from numpy.random import randint
import numpy as np
import torch

import os
import json


# City-wise Datamodule, contains the image embeddings (common to all partitions)
# and all the required partitions (train, train+val, val, test)
#
# In the Deep Sets variant, users are represented as a permutation-invariant set
# of image embeddings (up to K images) plus a binary mask that indicates which
# entries are real images and which are padding.


class ImageAuthorshipDataModule(LightningDataModule):
    def __init__(
        self,
        city,
        batch_size,
        num_workers=4,
        dataset_class=None,
        use_train_val=False,
        data_dir: Optional[Union[str, Path]] = None,
        limit_users: Optional[int] = None,
        max_user_images: int = 20,
        model_name: str = "PRESLEY",
        no_validation: bool = False,
    ) -> None:
        super().__init__()
        self.city = city
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Dataset class depends on the model family:
        # - MF_ELVis / ELVis -> BCE dataset
        # - PRESLEY          -> BPR dataset
        # - COLLEI           -> Contrastive dataset
        self.dataset_class = (
            TripadvisorImageAuthorshipBCEDataset
            if dataset_class is None
            else dataset_class
        )

        # If True, TRAIN_DEV is used as training set (instead of TRAIN).
        self.use_train_val = use_train_val

        # Data root directory (defaults to ./data)
        self.data_dir = Path(data_dir).expanduser() if data_dir is not None else Path("data")

        # Optional: limit to a fixed subset of users (useful for fast ablations / debugging)
        self.limit_users = limit_users
        self.user_subset_ids = None

        # Deep Sets: maximum number of images used to represent a user (K)
        self.max_user_images = max_user_images

        # Model name stored for logging / debugging if needed
        self.model_name = model_name

        # If True, skip validation set creation and val dataloader
        self.no_validation = no_validation

    def _build_user_history(self, partition_name: str):
        # Build user -> list[image_id] mapping from the selected partition.
        # Only positive samples are kept (take == 1), and duplicates are removed.
        partition_path = self.data_dir / self.city / "data_10+10" / f"{partition_name}_IMG"
        with partition_path.open("rb") as fp:
            df = pickle.load(fp)

        # Keep only the selected subset of users (if any)
        if self.user_subset_ids is not None:
            df = df[df["id_user"].isin(self.user_subset_ids)].reset_index(drop=True)

        take_col = "take"  # in TRAIN/TRAIN_DEV label column is "take"
        pos = (
            df[df[take_col] == 1]
            .drop_duplicates(subset=["id_user", "id_img"])   # avoid repeated (user, img)
            .sort_values(["id_user", "id_img"])
        )

        # Mapping {id_user: [id_img1, id_img2, ...]}
        user_to_ids = pos.groupby("id_user")["id_img"].apply(list).to_dict()

        return user_to_ids

    def setup(self, stage=None):
        # Make setup idempotent: Lightning may call it more than once.
        if getattr(self, "_is_setup_done", False):
            return
        self._is_setup_done = True

        # Load precomputed image embeddings (IMG_VEC): shape (num_images, 1536)
        embeddings_path = (
            self.data_dir / self.city / "data_10+10" / "IMG_VEC"
        )
        with embeddings_path.open("rb") as fp:
            self.image_embeddings = Tensor(pickle.load(fp))

        # If requested, compute/load a fixed subset of users for the experiment
        if self.limit_users:
            loaded = self._load_user_subset()

            if loaded is not None:
                self.user_subset_ids = loaded
            else:
                self.user_subset_ids = self._compute_user_subset()
                self._save_user_subset(self.user_subset_ids)

        # Deep Sets: build user history strictly from TRAIN (or TRAIN_DEV if use_train_val)
        # This avoids leakage from DEV/TEST when constructing the user representation.
        self.user_to_image_ids_train = self._build_user_history(
            partition_name="TRAIN" if not self.use_train_val else "TRAIN_DEV"
        )
        lens = pd.Series([len(v) for v in self.user_to_image_ids_train.values()])
        # print("[HIST] n_users:", len(lens))
        # print("[HIST] min/median/mean/max:", lens.min(), lens.median(), lens.mean(), lens.max())
        # print("[HIST] <=1:", (lens <= 1).mean(), " <=5:", (lens <= 5).mean(), " <=10:", (lens <= 10).mean())

        # Partitions
        self.train_dataset = self._get_dataset("TRAIN" if not self.use_train_val else "TRAIN_DEV")
        self.train_val_dataset = self._get_dataset("TRAIN_DEV")

        # Validation can be disabled (train-only runs)
        if self.no_validation:
            self.val_dataset = None
        else:
            self.val_dataset = self._get_dataset(
                "DEV" if not self.use_train_val else "TEST", set_type="validation"
            )

        self.test_dataset = self._get_dataset("TEST", set_type="test")

        print(
            f"{self.city:<10} | {self.train_dataset.nusers} users | {len(self.image_embeddings)} images"
        )

        self.nusers = self.train_dataset.nusers

    def _get_dataset(self, set_name, set_type="train"):
        # Factory for datasets; dataset_class is injected from utils.get_dataset_constructor(...)
        return self.dataset_class(
            datamodule=self,
            city=self.city,
            partition_name=set_name,
            set_type=set_type,
            user_subset_ids=self.user_subset_ids,
        )

    def _compute_user_subset(self):
        # Compute a deterministic subset of users based on number of positive samples in TRAIN/TRAIN_DEV.
        partition_name = "TRAIN" if not self.use_train_val else "TRAIN_DEV"
        partition_path = self.data_dir / self.city / "data_10+10" / f"{partition_name}_IMG"
        with partition_path.open("rb") as fp:
            df = pickle.load(fp)

        take_col = "take"  # in TRAIN/TRAIN_DEV label column is take
        pos = df[df[take_col] == 1]

        counts = pos.groupby("id_user").size().sort_values(ascending=False)

        # Minimum number of positive samples to be eligible
        min_pos = 10  # set to 5 if you run out of users
        eligible = counts[counts >= min_pos].index.to_list()

        subset = eligible[: self.limit_users]
        # print(f"Limiting to {len(subset)} users with >= {min_pos} positives (requested {self.limit_users})")
        return subset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            #persistent_workers=True,
        )

    def val_dataloader(self):
        # Lightning expects either a DataLoader or an empty list (to disable validation).
        if self.no_validation:
            return []
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            #persistent_workers=True,
        )

    def _subset_path(self):
        # Cache file for the selected subset of users (reproducible experiment runs).
        # Stored by city and training split mode.
        fname = f"user_subset_{self.limit_users}_trainval_{int(self.use_train_val)}.json"
        return os.path.join("models", self.city, fname)

    def _save_user_subset(self, user_ids):
        path = self._subset_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(map(int, user_ids)), f)
        # print(f"[SUBSET] saved {len(user_ids)} users -> {path}")

    def _load_user_subset(self):
        path = self._subset_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                user_ids = json.load(f)
            # print(f"[SUBSET] loaded {len(user_ids)} users <- {path}")
            return set(map(int, user_ids))
        return None

    def predict_dataloader(self):
        # In this project, predictions are computed over TEST by default.
        return self.test_dataloader()


# Dataset to train with BCE criterion
# Compatible with models: MF_ELVis,  ELVis
class TripadvisorImageAuthorshipBCEDataset(Dataset):
    def __init__(
        self,
        datamodule: ImageAuthorshipDataModule,
        city=None,
        partition_name=None,
        set_type="train",
        user_subset_ids=None,
    ):
        self.set_type = set_type
        self.city = city
        self.datamodule = datamodule
        self.partition_name = partition_name
        self.user_subset_ids = user_subset_ids

        # Name of the column that indicates sample label varies between partitions:
        # - TRAIN / TRAIN_DEV -> "take"
        # - DEV / TEST        -> "is_dev"
        self.takeordev = "is_dev" if partition_name in ["DEV", "TEST"] else "take"

        partition_path = (
            self.datamodule.data_dir
            / city
            / "data_10+10"
            / f"{partition_name}_IMG"
        )
        with partition_path.open("rb") as fp:
            self.dataframe = pickle.load(fp)

        # Apply subset restriction if enabled
        if self.user_subset_ids is not None:
            self.dataframe = self.dataframe[
                self.dataframe["id_user"].isin(self.user_subset_ids)
            ].reset_index(drop=True)

        self.nusers = self.dataframe["id_user"].nunique()

        print(
            f"{self.set_type} partition ({self.partition_name}_IMG)   | {len(self.dataframe)} samples | {self.nusers} users"
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Sample format:
        # - train/test      -> (user_id, image_embedding, target)
        # - validation      -> (user_id, image_embedding, target, id_test)
        user_id = self.dataframe.at[idx, "id_user"]
        image_id = self.dataframe.at[idx, "id_img"]
        image = self.datamodule.image_embeddings[image_id]

        target = float(self.dataframe.at[idx, self.takeordev])

        if self.set_type == "train" or self.set_type == "test":
            return user_id, image, target

        elif self.set_type == "validation":
            id_test = self.dataframe.at[idx, "id_test"]
            return user_id, image, target, id_test


# Dataset to train with BPR criterion
# Compatible with models: PRESLEY
class TripadvisorImageAuthorshipBPRDataset(TripadvisorImageAuthorshipBCEDataset):
    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipBPRDataset, self).__init__(**kwargs)
        self.max_user_images = getattr(self.datamodule, "max_user_images", 20)
        self.embedding_dim = int(self.datamodule.image_embeddings.shape[1])

        # Only build BPR triplets for TRAIN
        if self.set_type == "train":
            self._setup_bpr_dataframe()

    # Creates the BPR criterion samples with the form (user, positive image, negative image)
    def _setup_bpr_dataframe(self):
        # Separate between positive and negative samples
        self.positive_samples = (
            self.dataframe[self.dataframe[self.takeordev] == 1]
            .drop_duplicates(subset=["id_user", "id_img"])
            .sort_values(["id_user", "id_img"])
            .rename(columns={"id_img": "id_pos_img"})
            .reset_index(drop=True)
        )
        # self.positive_samples = self.positive_samples.drop_duplicates(
        #     keep='first').reset_index(drop=True)

        # Initialize neg sampling for this epoch
        self._resample_dataframe()

        # print("[BPR] positives:", len(self.positive_samples))
        # print("[BPR] bpr_dataframe:", len(self.bpr_dataframe))
        # print("[BPR] batches/epoch:", int(np.ceil(len(self.bpr_dataframe) / self.datamodule.batch_size)))

        # Diccionario: {user_id: [img_id1, img_id2, ..., img_idN]} y ordenado
        # self.user_to_image_ids = (
        #     self.positive_samples.sort_values(["id_user", "id_pos_img"])
        #     .groupby("id_user")["id_pos_img"]
        #     .apply(list)
        #     .to_dict()
        # )

        # hist = self.datamodule.user_to_image_ids_train
        # u = next(iter(hist.keys()))
        # print("DEBUG hist_train user", u, "n_ids", len(hist[u]))

    def _resample_dataframe(self):
        # Resample negative images for each (user, pos_img) to avoid overfitting.
        # Two kinds of negatives are used:
        # - different restaurant
        # - same restaurant (when possible)
        num_samples = len(self.positive_samples)

        same_res_bpr_samples = self.positive_samples.copy()
        different_res_bpr_samples = self.positive_samples.copy()

        # 1. Select images not from U and not from the same restaurant
        user_ids = self.positive_samples["id_user"].to_numpy()[:, None]
        img_ids = self.positive_samples["id_pos_img"].to_numpy()[:, None]
        rest_ids = self.positive_samples["id_restaurant"].to_numpy()[:, None]

        # List of the sample no. of the new neg_img of each BPR sample
        new_negatives = randint(num_samples, size=num_samples)

        # Count how many would have the same user in the neg_img and the pos_img
        num_invalid_samples = np.sum(
            (
                (user_ids[new_negatives] == user_ids)
                | (rest_ids[new_negatives] == rest_ids)
            )
        )
        while num_invalid_samples > 0:
            # Resample again the neg images for those samples until all are valid:
            # user(pos_img) != user(neg_img) and restaurant(pos_img) != restaurant(neg_img)
            new_negatives[
                np.where(
                    (
                        (user_ids[new_negatives] == user_ids)
                        | (rest_ids[new_negatives] == rest_ids)
                    )
                )[0]
            ] = randint(num_samples, size=num_invalid_samples)

            num_invalid_samples = np.sum(
                (
                    (user_ids[new_negatives] == user_ids)
                    | (rest_ids[new_negatives] == rest_ids)
                )
            )

        # Assign as new neg imgs the img_ids of the selected neg_imgs
        different_res_bpr_samples["id_neg_img"] = img_ids[new_negatives].squeeze(1)

        # 2. Select images not from U but from the same restaurant as the positive
        def obtain_samerest_samples(rest):
            # Restaurant-wise resampling (ensures different users inside the same restaurant group).
            user_ids = rest["id_user"].to_numpy()[:, None]
            img_ids = rest["id_pos_img"].to_numpy()[:, None]

            new_negatives = randint(len(rest), size=len(rest))
            num_invalid_samples = np.sum(user_ids[new_negatives] == user_ids)
            while num_invalid_samples > 0:
                new_negatives[np.where(user_ids[new_negatives] == user_ids)[0]] = (
                    randint(len(rest), size=num_invalid_samples)
                )

                num_invalid_samples = np.sum(user_ids[new_negatives] == user_ids)
            rest["id_neg_img"] = img_ids[new_negatives]

            return rest

        # Can't select "same restaurant" negatives if all photos of that restaurant are by the same user
        same_res_bpr_samples = (
            same_res_bpr_samples.groupby("id_restaurant")
            .filter(lambda g: g["id_user"].nunique() > 1)
            .reset_index(drop=True)
        )
        same_res_bpr_samples = (
            same_res_bpr_samples.groupby("id_restaurant", group_keys=False)
            .apply(obtain_samerest_samples)
            .reset_index(drop=True)
        )

        # Final BPR dataframe (concatenate both negative strategies)
        self.bpr_dataframe = pd.concat(
            [different_res_bpr_samples, same_res_bpr_samples], axis=0, ignore_index=True
        )

    def __len__(self):
        return (
            len(self.bpr_dataframe) if self.set_type == "train" else len(self.dataframe)
        )

    def __getitem__(self, idx):
        # If on training, return BPR samples
        if self.set_type == "train":
            user_id = self.bpr_dataframe.at[idx, "id_user"]
            neg_image_id = int(self.bpr_dataframe.at[idx, "id_neg_img"])
            pos_image_id = int(self.bpr_dataframe.at[idx, "id_pos_img"])

            # Positive and negative image embeddings (1536,)
            pos_image = self.datamodule.image_embeddings[pos_image_id]
            neg_image = self.datamodule.image_embeddings[neg_image_id]

            # Deep Sets: build (K, 1536) user set + (K,) mask from TRAIN history
            user_images, user_masks = self._build_user_representation(user_id, exclude_id=pos_image_id)

            return user_images, user_masks, pos_image, neg_image
            #return user_id, pos_image, neg_image  

        # If on validation, return samples
        # The test_id is needed to compute the validation recall or AUC
        # inside the LightningModule     
        elif self.set_type == "validation":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]
            
            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, "id_test"]

            # Deep Sets: build user representation excluding the evaluated image_id
            user_images, user_masks = self._build_user_representation(user_id, exclude_id=image_id)

            return user_images, user_masks, image, target, test_id
            #return user_id, image, target, test_id

        # If on test, return samples
        elif self.set_type == "test":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])

            # Deep Sets: build user representation excluding the evaluated image_id
            user_images, user_masks = self._build_user_representation(user_id, exclude_id=image_id)

            return user_images, user_masks, image, target
            #return user_id, image, target

    def _build_user_representation(self, user_id, exclude_id=None): # ONLY FOR DEEPSETS
        # Build a fixed-size set of user images (K, 1536) plus a mask (K,).
        # The user history is taken from TRAIN (or TRAIN_DEV if use_train_val).
        hist = self.datamodule.user_to_image_ids_train.get(user_id, []) # historial solo de TRAIN para ese usuario. Si no existe, []

        # 1) Construyo hist_excl 
        if exclude_id is None: # Si estoy evaluando una imagen, la quito del historial.
            hist_excl = list(hist)
        else:
            hist_excl = [i for i in hist if i != exclude_id]

        # 2) Fallback SOLO en train: si al excluir me quedo vacío, NO excluyo
        triggered_fallback = (
            self.set_type == "train"
            and exclude_id is not None
            and len(hist_excl) == 0
        )
        if triggered_fallback: # en no me quedo con el set vacío.
            hist_excl = list(hist)

        # Debug del fallback (solo 5 primeras)
        if triggered_fallback:
            if not hasattr(self, "_fallback_count"):
                self._fallback_count = 0
            self._fallback_count += 1
            # if self._fallback_count <= 5:
                # print(f"[FALLBACK] user {user_id} exclude_id {exclude_id} hist_len {len(hist)} -> using unexcluded hist")

        # 3) Selección de hasta K imágenes para el set del usuario
        if self.set_type == "train": # para TRAIN: sample aleatorio para no quedarme siempre con "las primeras K"
            if len(hist_excl) > self.max_user_images: # si hay muchas → sample aleatorio sin reemplazo (regulariza).
                image_ids = np.random.choice(hist_excl, size=self.max_user_images, replace=False).tolist()
                image_ids = [int(x) for x in image_ids] # np.random.choice devuelve np.int64; convierto a int por seguridad
            else: # si hay pocas → todas.
                image_ids = list(hist_excl)
        else: # para VAL/TEST: determinista (primeras K) para estabilidad de métrica
            image_ids = hist_excl[: self.max_user_images]

        # Prepare padding tensors with the same dtype/device as the stored embeddings.
        embeddings = self.datamodule.image_embeddings
        dtype = embeddings.dtype
        device = embeddings.device

        # 4) Embeddings + padding
        pad_tensor = torch.zeros(self.embedding_dim, dtype=dtype, device=device) # Vector 0 de tamaño 1536 para padding

        if len(image_ids) == 0: # Si no hay historial → todo padding.
            # Puede pasar en val/test si un user no tiene hist en train.
            image_tensors = [pad_tensor.clone() for _ in range(self.max_user_images)]
        else: # Si hay historial: cojo embeddings reales y rello hasta K con ceros.
            image_tensors = [embeddings[i] for i in image_ids]
            pad_len = self.max_user_images - len(image_tensors)
            if pad_len > 0:
                image_tensors.extend(pad_tensor.clone() for _ in range(pad_len))

        user_images = torch.stack(image_tensors) # FINAL USER (K, 1536)

        user_masks = torch.zeros(self.max_user_images, dtype=torch.float32, device=device) # FINAL MASK (K,)
        user_masks[: len(image_ids)] = 1.0 # 1 para reales, 0 para padding.
        
        return user_images, user_masks


# Dataset to train with Contrastive Loss criterion
# Compatible with models: COLLEI
class TripadvisorImageAuthorshipCLDataset(TripadvisorImageAuthorshipBCEDataset):
    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipCLDataset, self).__init__(**kwargs)
        self._setup_contrastive_learning_samples()

    # Creates the BPR criterion samples with the form (user, positive image, negative image)
    def _setup_contrastive_learning_samples(self):
        # Filter only positive samples for Contrastive Learning
        self.pos_dataframe = (
            self.dataframe[self.dataframe[self.takeordev] == 1]
            .drop_duplicates(keep="first")
            .reset_index(drop=True)
        )

    def __len__(self):
        return (
            len(self.pos_dataframe) if self.set_type == "train" else len(self.dataframe)
        )

    def __getitem__(self, idx):
        # If on training or validation, return CL samples
        # (user, pos_image)
        if self.set_type == "train":
            user_id = self.pos_dataframe.at[idx, "id_user"]

            image_id = self.pos_dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            return user_id, image

        # If on test, return normal samples
        # (user, image, label)
        elif self.set_type == "validation":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, "id_test"]

            return user_id, image, target, test_id

        elif self.set_type == "test":
            user_id = self.dataframe.at[idx, "id_user"]

            image_id = self.dataframe.at[idx, "id_img"]
            image = self.datamodule.image_embeddings[image_id]

            target = float(self.dataframe.at[idx, self.takeordev])

            return user_id, image, target
