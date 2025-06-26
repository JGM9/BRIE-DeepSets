



import pickle
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from pytorch_lightning import LightningDataModule
import pandas as pd
from numpy.random import randint
import numpy as np
import torch


# City-wise Datamodule, contiene los embeddings de imágenes (común a todas las particiones)
# y todas las particiones necesarias (train, train+val, val, test)
class ImageAuthorshipDataModule(LightningDataModule):
    def __init__(
        self, city, batch_size, num_workers=4, dataset_class=None, use_train_val=False
    ) -> None:
        super().__init__()
        self.city = city
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_class = (
            TripadvisorImageAuthorshipBCEDataset
            if dataset_class is None
            else dataset_class
        )
        self.use_train_val = use_train_val

        # Llamamos a setup() para cargar datos
        self.setup()

    def setup(self, stage=None):
        # Cargamos únicamente el tensor de embeddings de imágenes
        self.image_embeddings = Tensor(
            pickle.load(
                open(
                    "C:/Users/Usuario/Desktop/cuarto/2o_cuatrimestre/TFG/BRIE-MASTER/data/"
                    + self.city
                    + "/data_10+10/IMG_VEC",
                    "rb",
                )
            )
        )

        # Creamos cada partición: TRAIN o TRAIN_DEV (si usamos train+val), etc.
        self.train_dataset = self._get_dataset(
            "TRAIN" if not self.use_train_val else "TRAIN_DEV"
        )
        self.train_val_dataset = self._get_dataset("TRAIN_DEV")
        self.val_dataset = self._get_dataset(
            "DEV" if not self.use_train_val else "TEST", set_type="validation"
        )
        self.test_dataset = self._get_dataset("TEST", set_type="test")

        # Obtenemos nusers desde el dataset de entrenamiento
        self.nusers = self.train_dataset.nusers

        print(
            f"{self.city:<10} | {self.nusers} users | {len(self.image_embeddings)} images"
        )

    def _get_dataset(self, set_name, set_type="train"):
        # Pasamos self (con image_embeddings) al dataset, pero este lo guardará en self.image_embeddings
        return self.dataset_class(
            datamodule=self, city=self.city, partition_name=set_name, set_type=set_type
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


# Dataset para entrenar con BCE (Modelos: MF_ELVis, ELVis)
class TripadvisorImageAuthorshipBCEDataset(Dataset):
    def __init__(
        self,
        datamodule: ImageAuthorshipDataModule,
        city=None,
        partition_name=None,
        set_type="train",
    ):
        self.set_type = set_type
        self.city = city

        # Guardamos solo el tensor de embeddings, no todo el datamodule
        self.image_embeddings = datamodule.image_embeddings

        self.partition_name = partition_name

        # La columna de etiqueta varía entre particiones: "take" o "is_dev"
        self.takeordev = "is_dev" if partition_name in ["DEV", "TEST"] else "take"

        # Cargamos el dataframe pickled de esta partición
        self.dataframe = pickle.load(
            open(
                f"C:/Users/Usuario/Desktop/cuarto/2o_cuatrimestre/TFG/BRIE-MASTER/data/{city}/data_10+10/{partition_name}_IMG",
                "rb",
            )
        )

        # nusers = número de usuarios únicos en este dataframe
        self.nusers = self.dataframe["id_user"].nunique()

        print(
            f"{self.set_type} partition ({self.partition_name}_IMG)   | {len(self.dataframe)} samples | {self.nusers} users"
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        user_id = self.dataframe.at[idx, "id_user"]
        image_id = self.dataframe.at[idx, "id_img"]
        # Obtenemos el embedding desde el tensor local
        image = self.image_embeddings[image_id]

        target = float(self.dataframe.at[idx, self.takeordev])

        if self.set_type in ("train", "test"):
            return user_id, image, target
        else:  # validation
            test_id = self.dataframe.at[idx, "id_test"]
            return user_id, image, target, test_id


# Dataset para entrenar con BPR (Modelos: PRESLEY)
class TripadvisorImageAuthorshipBPRDataset(TripadvisorImageAuthorshipBCEDataset):
    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipBPRDataset, self).__init__(**kwargs)
        # Construimos el dataframe BPR tras haber heredado imagenes y dataframe BCE
        self._setup_bpr_dataframe()

    def _setup_bpr_dataframe(self):
        # Tomamos solo las muestras positivas (takeordev == 1)
        self.positive_samples = (
            self.dataframe[self.dataframe[self.takeordev] == 1]
            .sort_values(["id_user", "id_img"])
            .rename(columns={"id_img": "id_pos_img"})
            .reset_index(drop=True)
        )

        # Generamos id_neg_img
        self._resample_dataframe()

        # Diccionario: {user_id: [img_id1, img_id2, ...]}
        self.user_to_image_ids = (
            self.dataframe
            .sort_values(["id_user", "id_img"])
            .groupby("id_user")["id_img"]
            .apply(list)
            .to_dict()
        )

        self.max_user_images = 20

    def _resample_dataframe(self):
        num_samples = len(self.positive_samples)

        same_res_bpr_samples = self.positive_samples.copy()
        different_res_bpr_samples = self.positive_samples.copy()

        # 1. Seleccionar imágenes de "diferente rest" y "diferente user"
        user_ids = self.positive_samples["id_user"].to_numpy()[:, None]
        img_ids = self.positive_samples["id_pos_img"].to_numpy()[:, None]
        rest_ids = self.positive_samples["id_pos_img"].to_numpy()[:, None]

        new_negatives = randint(num_samples, size=num_samples)

        num_invalid_samples = np.sum(
            (
                (user_ids[new_negatives] == user_ids)
                | (rest_ids[new_negatives] == rest_ids)
            )
        )
        while num_invalid_samples > 0:
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

        different_res_bpr_samples["id_neg_img"] = img_ids[new_negatives]

        # 2. Seleccionar imágenes de "mismo rest" pero "diferente user"
        def obtain_samerest_samples(rest):
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

        self.bpr_dataframe = pd.concat(
            [different_res_bpr_samples, same_res_bpr_samples], axis=0, ignore_index=True
        )

    def __len__(self):
        if self.set_type == "train":
            return len(self.bpr_dataframe)
        else:
            return len(self.dataframe)

    def __getitem__(self, idx):
        if self.set_type == "train":
            user_id = self.bpr_dataframe.at[idx, "id_user"]
            pos_image_id = self.bpr_dataframe.at[idx, "id_pos_img"]
            neg_image_id = self.bpr_dataframe.at[idx, "id_neg_img"]

            pos_image = self.image_embeddings[pos_image_id]
            neg_image = self.image_embeddings[neg_image_id]

            image_ids = self.user_to_image_ids[user_id][: self.max_user_images]
            image_tensors = [self.image_embeddings[i] for i in image_ids]

            pad_len = self.max_user_images - len(image_tensors)
            if pad_len > 0:
                pad_tensor = torch.zeros_like(image_tensors[0])
                image_tensors += [pad_tensor] * pad_len

            user_images = torch.stack(image_tensors)
            user_masks = torch.tensor([1] * len(image_ids) + [0] * pad_len, dtype=torch.float32)

            return user_images, user_masks, pos_image, neg_image

        elif self.set_type == "validation":
            user_id = self.dataframe.at[idx, "id_user"]
            image_id = self.dataframe.at[idx, "id_img"]
            image = self.image_embeddings[image_id]
            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, "id_test"]

            image_ids = self.user_to_image_ids[user_id][: self.max_user_images]
            image_tensors = [self.image_embeddings[i] for i in image_ids]
            pad_len = self.max_user_images - len(image_tensors)
            if pad_len > 0:
                pad_tensor = torch.zeros_like(image_tensors[0])
                image_tensors += [pad_tensor] * pad_len

            user_images = torch.stack(image_tensors)
            user_masks = torch.tensor([1] * len(image_ids) + [0] * pad_len, dtype=torch.float32)

            return user_images, user_masks, image, target, test_id

        else:  # test
            user_id = self.dataframe.at[idx, "id_user"]
            image_id = self.dataframe.at[idx, "id_img"]
            image = self.image_embeddings[image_id]
            target = float(self.dataframe.at[idx, self.takeordev])

            image_ids = self.user_to_image_ids[user_id][: self.max_user_images]
            image_tensors = [self.image_embeddings[i] for i in image_ids]
            pad_len = self.max_user_images - len(image_tensors)
            if pad_len > 0:
                pad_tensor = torch.zeros_like(image_tensors[0])
                image_tensors += [pad_tensor] * pad_len

            user_images = torch.stack(image_tensors)
            user_masks = torch.tensor([1] * len(image_ids) + [0] * pad_len, dtype=torch.float32)

            return user_images, user_masks, image, target


# Dataset para entrenar con Contrastive Loss (Modelos: COLLEI)
class TripadvisorImageAuthorshipCLDataset(TripadvisorImageAuthorshipBCEDataset):
    def __init__(self, **kwargs) -> None:
        super(TripadvisorImageAuthorshipCLDataset, self).__init__(**kwargs)
        self._setup_contrastive_learning_samples()

    def _setup_contrastive_learning_samples(self):
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
        if self.set_type == "train":
            user_id = self.pos_dataframe.at[idx, "id_user"]
            image_id = self.pos_dataframe.at[idx, "id_img"]
            image = self.image_embeddings[image_id]
            return user_id, image

        elif self.set_type == "validation":
            user_id = self.dataframe.at[idx, "id_user"]
            image_id = self.dataframe.at[idx, "id_img"]
            image = self.image_embeddings[image_id]
            target = float(self.dataframe.at[idx, self.takeordev])
            test_id = self.dataframe.at[idx, "id_test"]
            return user_id, image, target, test_id

        else:  # test
            user_id = self.dataframe.at[idx, "id_user"]
            image_id = self.dataframe.at[idx, "id_img"]
            image = self.image_embeddings[image_id]
            target = float(self.dataframe.at[idx, self.takeordev])
            return user_id, image, target
