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


## IMPLEMENTACION DE LOS DEEP SETS:

from torch import nn, Tensor
import torch

class DeepSetEmbeddingBlock(nn.Module): # u = ρ((Σ_i m_i · φ(x_i))
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

        # phi: per-element encoder
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

        # rho: set-level refinement (input dim = d siempre)
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

        # image projection
        self.img_fc = nn.Linear(1536, d)


    def _log_stats(self, name: str, t: Tensor): # debuggear las estadisticas detectando colapsos de embeddings.
        self._dbg_step += 1
        if (self._dbg_step % self.debug_every) != 0:
            return
        # with torch.no_grad():
        #     mean = t.mean().item()
        #     std = t.std(unbiased=False).item()
        #     norm = t.norm(dim=-1).mean().item() if t.dim() >= 2 else t.norm().item()
        #print(f"[SANITY] {name} mean/std/norm: {mean:.4f}/{std:.4f}/{norm:.4f}")

    def encode_user(self, user_images: Tensor, user_masks: Tensor) -> Tensor:
        if self.debug_sanity: # Sanity checks (B = batch size & K = max number of images per user)
            assert user_images.dim() == 3, f"user_images dim expected 3, got {user_images.shape}" # user_images: (B, K, 1536)
            assert user_masks.dim() == 2, f"user_masks dim expected 2, got {user_masks.shape}" # user_masks: (B, K)
            assert user_images.size(0) == user_masks.size(0), "batch mismatch" # coherencia entre las imagenes y la representacion de los usuarios
            assert user_images.size(1) == user_masks.size(1), "num images mismatch"

        user_masks = user_masks.to(user_images.device).float() # (B, K)

        phi_x = self.phi(user_images) # Aplicar phi φ a las imagenes (B, K, d)
        if self.debug_sanity:
            self._log_stats("phi_out", phi_x)

        phi_masked = phi_x * user_masks.unsqueeze(-1) # Anula las imágenes de padding (B, K, d)
        mask_sum = user_masks.sum(dim=1, keepdim=True) # Número real de imágenes por usuario (B, 1)
        mask_sum_clamped = mask_sum.clamp(min=1.0).to(phi_masked.dtype) # Evita división por 0
        aggregated = phi_masked.sum(dim=1) / mask_sum_clamped  # Media del conjunto (B, d)

        if self.debug_sanity:
            self._log_stats("pooled", aggregated)
            # print(f"[SANITY] mask_sum min/mean/max: {mask_sum.min().item():.2f}/{mask_sum.mean().item():.2f}/{mask_sum.max().item():.2f}")

        if not self.use_rho:
            return aggregated

        out = self.rho(aggregated) # Refinamiento del embedding de usuario (B, d) => Lo que se devuelve
        if self.debug_sanity:
            self._log_stats("rho_out", out)
        return out

    def project_images(self, images: Tensor) -> Tensor: # Proyección de las imágenes
        return self.img_fc(images)

    def forward(self, user_images, user_masks, images):
        # Dos embeddings comparables: user y img
        u_embeddings = self.encode_user(user_images, user_masks)
        img_embeddings = self.project_images(images)
        return u_embeddings, img_embeddings
