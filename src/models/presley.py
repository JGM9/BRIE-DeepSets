import torchmetrics
import torch
from torch import nn
from src.models.losses import bpr_loss, UserwiseAUCROC
from src.models.mf_elvis import MF_ELVis
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
import math

from src.models.blocks import DeepSetEmbeddingBlock

class PRESLEY(MF_ELVis):
    def __init__(
        self,
        d: int,
        nusers: int,
        lr: float,
        phi=None,
        rho=None,
        dropout=0.0,
        debug_sanity: bool = False,
        debug_sanity_freq: int = 200,
        ds_no_rho: bool = False,
        debug_overfit: bool = False,
    ):
        """
        BPR (Bayesian Pairwise Ranking) loss based Matrix Factorisation model for image autorship

        Parameters:
            d: int
                Size of the latent image and user embeddings
            nusers: int
                Number of users in the dataset (used for the user embedding layer)
            lr: float
                Learning rate of the model
            dropout: float
                Training dropout of the image and user embeddings before inner product
        """

        super().__init__(d=d, nusers=nusers, lr=lr)

        if debug_overfit:
            dropout = 0.0

        self.embedding_block = DeepSetEmbeddingBlock(
            d,
            phi=phi,
            rho=rho,
            use_rho=not ds_no_rho,          # <-- CAMBIO AQUÍ
            debug_sanity=debug_sanity,
        )

        # print(
        #     f"[SANITY] DeepSets config | "
        #     f"use_rho={not ds_no_rho} | "   # <-- CAMBIO AQUÍ
        # )


        # Dropouts before dot product
        self.user_dropout = Dropout(dropout)
        self.img_dropout = Dropout(dropout)

        self._init_mlp_weights(self.embedding_block.phi)
        if self.embedding_block.use_rho:
            self._init_mlp_weights(self.embedding_block.rho)
        if isinstance(self.embedding_block.img_fc, nn.Linear):
            xavier_uniform_(self.embedding_block.img_fc.weight.data, gain=1.0)
            if self.embedding_block.img_fc.bias is not None:
                self.embedding_block.img_fc.bias.data.zero_()

        self.criterion = None  # Just to sanitize
        self.debug_sanity = debug_sanity
        self.debug_sanity_freq = debug_sanity_freq
        self.ds_no_rho = ds_no_rho
        self.save_hyperparameters(ignore=["phi", "rho"])

    def training_step(self, batch, batch_idx):
        user_images, user_masks, pos_images, neg_images = batch

        # if self.debug_sanity and batch_idx == 0:
        #     print(
        #         "[SANITY][device check]",
        #         "user_images:", user_images.device,
        #         "pos_images:", pos_images.device,
        #         "neg_images:", neg_images.device,
        #     )


        u_embeddings = self.embedding_block.encode_user(user_images, user_masks)
        pos_img_embeddings = self.embedding_block.project_images(pos_images)
        neg_img_embeddings = self.embedding_block.project_images(neg_images)

        if self.debug_sanity and self._should_log_sanity(batch_idx):
            self._log_sanity_stats(
                user_masks=user_masks,
                u_embeddings=u_embeddings,
                pos_embeddings=pos_img_embeddings,
                neg_embeddings=neg_img_embeddings,
            )

        u_embeddings = self.user_dropout(u_embeddings)
        pos_img_embeddings = self.img_dropout(pos_img_embeddings)
        neg_img_embeddings = self.img_dropout(neg_img_embeddings)

        pos_preds = torch.sum(u_embeddings * pos_img_embeddings, dim=-1)
        neg_preds = torch.sum(u_embeddings * neg_img_embeddings, dim=-1)

        diff = pos_preds - neg_preds

        #users, pos_images, neg_images = batch

        #pos_preds = self((users, pos_images), output_logits=True)
        #neg_preds = self((users, neg_images), output_logits=True)

        loss = bpr_loss(pos_preds, neg_preds)

        if self.debug_sanity and self._should_log_sanity(batch_idx, force_epoch=True):
            self._log_scores(
                pos_preds=pos_preds,
                neg_preds=neg_preds,
                diff=diff,
                user_masks=user_masks,
            )

        # Logging only for print purposes
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        user_images, user_masks, images, targets, id_tests = batch

        preds = self((user_images, user_masks, images), output_logits=True)

        #users, images, targets, id_tests = batch  
        
        #preds = self((users, images), output_logits=True)

        self.val_recall.update(preds, targets.long(), id_tests)
        self.val_auc.update(preds, targets.long(), id_tests)

        self.log(
            "val_recall", self.val_recall, on_epoch=True, logger=True, prog_bar=True
        )
        self.log("val_auc", self.val_auc, on_epoch=True, logger=True, prog_bar=True)

    def forward(self, x, output_logits=False):
        user_images, user_masks, images = x

        u_embeddings, img_embeddings = self.embedding_block(user_images, user_masks, images)

        if self.debug_sanity and not output_logits:
            self._log_sanity_stats(
                user_masks=user_masks,
                u_embeddings=u_embeddings,
                pos_embeddings=img_embeddings,
                neg_embeddings=None,
                is_eval=True,
            )

        if output_logits:
            u_embeddings = self.user_dropout(u_embeddings)
            img_embeddings = self.img_dropout(img_embeddings)

        # Using dim=-1 to support forward of batches and single samples
        preds = torch.sum(u_embeddings * img_embeddings, dim=-1)

        if not output_logits:
            preds = torch.sigmoid(preds)
        return preds

    def on_train_epoch_start(self) -> None:
        # Between epochs, resample the negative images of each (user, pos_img, neg_img)
        # sample tryad to avoid overfitting
        self.trainer.train_dataloader.dataset._resample_dataframe()

    @staticmethod
    def _init_mlp_weights(module: nn.Module) -> None:
        for submodule in module.modules():
            if isinstance(submodule, nn.Linear):
                xavier_uniform_(submodule.weight.data, gain=1.0)
                if submodule.bias is not None:
                    submodule.bias.data.zero_()

    def _should_log_sanity(self, batch_idx: int, force_epoch: bool = False) -> bool:
        if not self.debug_sanity:
            return False
        if force_epoch and batch_idx == 0:
            return True
        return (self.global_step % max(self.debug_sanity_freq, 1)) == 0

    def _log_sanity_stats(
        self,
        user_masks,
        u_embeddings,
        pos_embeddings,
        neg_embeddings=None,
        is_eval: bool = False,
    ):
        with torch.no_grad():
            mask_sum = user_masks.sum(dim=1)
            mask_min, mask_mean, mask_max = (
                mask_sum.min().item(),
                mask_sum.float().mean().item(),
                mask_sum.max().item(),
            )
            zero_masks = (mask_sum == 0).float().mean().item()

            u_mean = u_embeddings.mean().item()
            u_std = u_embeddings.std().item()
            u_norm = u_embeddings.norm(dim=-1).mean().item()

            log_lines = [
                f"[SANITY]{' [eval]' if is_eval else ''} mask_sum min/mean/max: {mask_min:.2f}/{mask_mean:.2f}/{mask_max:.2f}",
                f"[SANITY] zero-mask users: {100*zero_masks:.2f}%",
                f"[SANITY] U mean/std/norm: {u_mean:.4f}/{u_std:.4f}/{u_norm:.4f}",
            ]

            if pos_embeddings is not None:
                log_lines.append(
                    f"[SANITY] V_pos mean/std/norm: {pos_embeddings.mean().item():.4f}/"
                    f"{pos_embeddings.std().item():.4f}/{pos_embeddings.norm(dim=-1).mean().item():.4f}"
                )
            if neg_embeddings is not None:
                log_lines.append(
                    f"[SANITY] V_neg mean/std/norm: {neg_embeddings.mean().item():.4f}/"
                    f"{neg_embeddings.std().item():.4f}/{neg_embeddings.norm(dim=-1).mean().item():.4f}"
                )

            if zero_masks > 0:
                log_lines.append("[WARN] Some users have mask_sum == 0; padding may be wrong.")
            if abs(u_std) < 1e-6 or abs(u_norm) < 1e-6:
                log_lines.append("[WARN] User embeddings collapsing (std or norm ~0).")

            # print("\n".join(log_lines))

    def _log_scores(self, pos_preds, neg_preds, diff, user_masks):
        with torch.no_grad():
            diff_mean = diff.mean().item()
            diff_std = diff.std().item()
            pct_pos = (diff > 0).float().mean().item()
            log_lines = [
                f"[SANITY] score_pos mean/std: {pos_preds.mean().item():.4f}/{pos_preds.std().item():.4f}",
                f"[SANITY] score_neg mean/std: {neg_preds.mean().item():.4f}/{neg_preds.std().item():.4f}",
                f"[SANITY] diff mean/std: {diff_mean:.4f}/{diff_std:.4f}",
                f"[SANITY] pct diff>0: {pct_pos:.4f}",
            ]
            sig_pos = torch.sigmoid(pos_preds)
            sig_neg = torch.sigmoid(neg_preds)
            log_lines += [
            f"[SANITY] sigmoid(pos) mean/min/max: {sig_pos.mean().item():.4f}/{sig_pos.min().item():.4f}/{sig_pos.max().item():.4f}",
            f"[SANITY] sigmoid(neg) mean/min/max: {sig_neg.mean().item():.4f}/{sig_neg.min().item():.4f}/{sig_neg.max().item():.4f}",
            ]


            mask_sum = user_masks.sum(dim=1)
            if (mask_sum == 0).any():
                log_lines.append("[WARN] Detected users with mask_sum==0 during score logging.")
            if abs(diff_std) < 1e-6:
                log_lines.append("[WARN] score diff std ~0; model may not separate positives/negatives.")

            # print("\n".join(log_lines))

    def on_after_backward(self):
        if not self.debug_sanity:
            return
        if not self._should_log_sanity(batch_idx=0):
            return

        def grad_norm(module):
            if module is None:
                return None
            if hasattr(module, "weight"):
                grad = module.weight.grad
            else:
                grad = None
            return None if grad is None else grad.norm().item()

        def first_linear(module: nn.Module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    return m
            return None

        phi_layer = first_linear(self.embedding_block.phi)
        img_layer = (
            first_linear(self.embedding_block.img_fc)
            if isinstance(self.embedding_block.img_fc, nn.Module)
            else None
        )

        layers = [("phi", phi_layer), ("img_fc", img_layer)]
        if self.embedding_block.use_rho:
            layers.insert(1, ("rho", first_linear(self.embedding_block.rho)))
        grad_logs = []

        for name, layer in layers:
            norm_val = grad_norm(layer)
            if norm_val is None:
                grad_logs.append(f"[WARN] No gradient for {name} layer")
            else:
                grad_logs.append(f"[SANITY] grad norm {name}: {norm_val:.6f}")
                if norm_val < 1e-8:
                    grad_logs.append(f"[WARN] Gradient norm for {name} ~0")

        # print("\n".join(grad_logs))
        
