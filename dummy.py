import torch, torch.nn as nn

torch.manual_seed(0)

B, M, Din, H, D = 4, 5, 1536, 128, 64  # batch, imágenes por usuario, dims
X = torch.randn(B, M, Din)             # embeddings de fotos del usuario (dummy)
mask = torch.tensor([[1,1,1,0,0],
                     [1,1,0,0,0],
                     [1,1,1,1,1],
                     [1,0,0,0,0]], dtype=torch.float32)  # máscaras

# φ y ρ (versión mínima como en tu modelo)
phi = nn.Linear(Din, H, bias=False)
rho = nn.Linear(H, D, bias=False)

# Agregación: media enmascarada (equivalente a suma si divides por #activos)
phi_X = phi(X)                                          # [B,M,H]
sum_pool = (phi_X * mask.unsqueeze(-1)).sum(dim=1)      # [B,H]
den = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)      # evitar /0
Z = sum_pool / den                                      # [B,H]
U = rho(Z)                                              # [B,D] → representación usuario

# Invariancia a permutación
perm = torch.randperm(M)
phi_X_perm = phi(X[:, perm, :])
sum_pool_perm = (phi_X_perm * mask[:, perm].unsqueeze(-1)).sum(1)
Z_perm = sum_pool_perm / den
U_perm = rho(Z_perm)
assert torch.allclose(U, U_perm, atol=1e-6), "No invariante → revisar φ/ρ o máscaras"

# Gradiente: que todo sea diferenciable
V = torch.randn(B, Din)                  # embedding de foto (dummy)
P = nn.Linear(Din, D, bias=False)        # proyección foto 1536→d (como en PRESLEY)
Vp = P(V)                                # [B,D]
scores = (U * Vp).sum(dim=1).mean()      # ⟨U, V′⟩ promedio
scores.backward()
for name, p in [("phi", phi.weight), ("rho", rho.weight), ("P", P.weight)]:
    assert p.grad is not None and torch.isfinite(p.grad).all(), f"Gradiente nulo/NaN en {name}"

print("OK: φ/ρ invariante con máscara, shapes correctos y gradientes fluyen.")
