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

# Definimos el bloque DeepSet como un módulo de PyTorch
class DeepSetEmbeddingBlock(nn.Module):
    def __init__(self, d, phi=None):
        """
        Args:
            d (int): Dimensión final de embedding (= 'd' en el original).
            phi (nn.Module): modelo CNN que usaremos para procesar cada imagen del usuario.
        """
        super().__init__()  
        self.d = d  # dimensión de salida deseada para los embeddings de usuario
        # MODELO PARA USER
        self.rho = nn.Identity()
        self.phi = nn.Linear(1536, d) 

        # MODELO PARA IMAGEN
        self.img_fc = nn.Linear(1536, d)  # capa lineal para mapear el embedding de la foto objetivo (1536) a dimensión d


    def forward(self, user_images, user_masks, images):
        """
        user_images: (batch_size, num_user_images, Tamaño_embedding = 1536)
        user_masks:  (batch_size, num_user_images), valores 1 para imágenes válidas, 0 para relleno
        images:      (batch_size, Tamaño_embedding = 1536) embedding de la imagen candidata 
        """
        # PARTE DE USER:
        B, N, T = user_images.shape  # Desempaquetamos batch_size (B), número máximo de fotos por usuario (N) y dimensión T

        phi_x = self.phi(user_images)  # Aplicamos φ a cada embedding de imagen: resultado (B, N, T)

        phi_mask = phi_x * user_masks.unsqueeze(-1)  # Multiplicamos por la máscara (B, N, 1) para “anular” las posiciones 0000

        # Sumamos por filas (las N imágenes) / número real de fotos de cada usuario => media de los embeddings, forma (B, 1536)
        counts = user_masks.sum(dim=1, keepdim=True).clamp_min(1)  # (batch_size, 1)
        aggregated = phi_mask.sum(dim=1) / counts                  # (batch_size, 1536)

        u_embeddings = self.rho(aggregated)  # Aplicamos ρ a aggregated => embedding final de usuario (B, d)

        # PARTE DE IMAGEN:
        img_embeddings = self.img_fc(images) # Aplicamos la capa de la imagen candidata (B, d)

        return u_embeddings, img_embeddings  