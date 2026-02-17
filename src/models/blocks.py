from torch import nn, Tensor
import torch

# Embedding Block for image authorship task
# Inputs: user id, pre-trained image embedding
# Outputs (batch_size x d) user embedding, (batch_size x d) image embedding


class ImageAutorshipEmbeddingBlock(nn.Module):
    def __init__(self, d, nusers, initrange=0.05):
        super().__init__()
        self.d = d
        self.nusers = nusers
        self.u_emb = nn.Embedding(num_embeddings=nusers, embedding_dim=d)
        self.img_fc = nn.Linear(1536, d)
        # self._init_weights(initrange=initrange)  # UNCOMMENT THIS TO INIT WEIGHTS

    def _init_weights(self, initrange=0.01):
        self.u_emb.weight.data.uniform_(-initrange, initrange)
        self.img_fc.weight.data.uniform_(-initrange, initrange)
        self.img_fc.bias.data.zero_()

    def forward(self, users, images):
        # Ensure we work with tensors in the case of single sample inputs
        if not torch.is_tensor(users):
            users = torch.tensor(users, dtype=torch.int32)

        u_embeddings = self.u_emb(users)
        img_embeddings = self.img_fc(images)

        return u_embeddings, img_embeddings


# Deep Sets embedding block
# User representation is computed as:
#   u = ρ( mean_i ( φ(x_i) ) )
# where x_i are the image embeddings belonging to the user.

class DeepSetEmbeddingBlock(nn.Module):
    def __init__(
        self,
        d: int,
        phi: nn.Module = None,
        rho: nn.Module = None,
        use_rho: bool = True,
        debug_sanity: bool = False,
    ):
        super().__init__()
        self.d = d
        self.use_rho = use_rho
        self.debug_sanity = debug_sanity
        self.debug_every = 50
        self._dbg_step = 0

        # φ: element-wise encoder applied to each image
        self.phi = (
            phi
            if phi is not None
            else nn.Sequential(
                nn.Linear(1536, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )
        )

        # ρ: optional set-level refinement after aggregation
        self.rho = (
            rho
            if rho is not None
            else nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(inplace=True),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )
        )

        # Image projection for candidate items
        self.img_fc = nn.Linear(1536, d)

    def _log_stats(self, name: str, t: Tensor):
        self._dbg_step += 1
        if (self._dbg_step % self.debug_every) != 0:
            return

    def encode_user(self, user_images: Tensor, user_masks: Tensor) -> Tensor:
        # user_images: (B, K, 1536)
        # user_masks:  (B, K)

        if self.debug_sanity:
            assert user_images.dim() == 3, f"user_images dim expected 3, got {user_images.shape}"
            assert user_masks.dim() == 2, f"user_masks dim expected 2, got {user_masks.shape}"
            assert user_images.size(0) == user_masks.size(0), "batch mismatch"
            assert user_images.size(1) == user_masks.size(1), "num images mismatch"

        user_masks = user_masks.to(user_images.device).float()

        # Apply φ to each image
        phi_x = self.phi(user_images)
        if self.debug_sanity:
            self._log_stats("phi_out", phi_x)

        # Mask padded images
        phi_masked = phi_x * user_masks.unsqueeze(-1)

        # Mean aggregation over the set
        mask_sum = user_masks.sum(dim=1, keepdim=True)
        mask_sum_clamped = mask_sum.clamp(min=1.0).to(phi_masked.dtype)
        aggregated = phi_masked.sum(dim=1) / mask_sum_clamped

        if self.debug_sanity:
            self._log_stats("pooled", aggregated)

        if not self.use_rho:
            return aggregated

        # Apply ρ refinement
        out = self.rho(aggregated)
        if self.debug_sanity:
            self._log_stats("rho_out", out)

        return out

    def project_images(self, images: Tensor) -> Tensor:
        # Linear projection of candidate images
        return self.img_fc(images)

    def forward(self, user_images, user_masks, images):
        # Produce comparable user and image embeddings
        u_embeddings = self.encode_user(user_images, user_masks)
        img_embeddings = self.project_images(images)
        return u_embeddings, img_embeddings
