from time import time
import pytorch_lightning as pl
from torch import optim
import torchmetrics

from src.models.losses import UserwiseAUCROC


class BaseModelForImageAuthorship(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = kwargs["lr"]

        self.val_recall = torchmetrics.RetrievalRecall(k=10)
        self.val_auc = UserwiseAUCROC()
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")

    def forward(self, x):
        return NotImplementedError

    def training_step(self, batch, batch_idx):
        return NotImplementedError

    def validation_step(self, batch, batch_idx):
        return NotImplementedError

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        user_images, user_masks, images, _ = batch
        #users, images, _ = batch
        #return self((users, images))
        return self((user_images, user_masks, images))
    
    def on_test_epoch_end(self) -> None:
        return

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
