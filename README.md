# BRIE-DeepSets: Scalable User Representation with Deep Sets for Visual Explainability

**Authors (original BRIE):**  
Jorge Paz-Ruza, Amparo Alonso-Betanzos  
Berta Guijarro-Berdiñas, Brais Cancela, Carlos Eiras-Franco  

**Extension and implementation:**  
**Jorge García Mateo**

---

## 1. Overview

This repository presents **BRIE-DeepSets**, an extension of the BRIE model for visual explainability in recommender systems, where the traditional user representation based on fixed user identifiers is replaced by a **set-based representation built from user images using Deep Sets**.

In the original BRIE model, each user is associated with a learned embedding indexed by a unique identifier. While effective, this approach limits scalability, as the model must be retrained whenever new users are added.  
BRIE-DeepSets removes this limitation by representing users as permutation-invariant sets of visual embeddings, allowing the model to generalize naturally to unseen users without modifying the architecture or retraining user-specific parameters.

The proposed approach preserves the Bayesian Pairwise Ranking (BPR) objective and the explainability-oriented design of BRIE, while improving scalability and flexibility for real-world deployment.

---

## 2. Key Contributions

- Replacement of fixed user-ID embeddings with **Deep Sets–based user representations**
- Permutation-invariant modeling of user histories using visual content
- Seamless integration with the original BRIE training and evaluation pipeline
- Empirical evaluation across six real-world datasets
- Analysis of the trade-off between scalability and ranking performance

---
## 3. Architecture

### 3.1 User Representation with Deep Sets

Each user is represented as a set of visual embeddings extracted from images previously uploaded by that user.

The user embedding is computed following the Deep Sets formulation:

$$
u = \rho \left( \sum_{i=1}^{N_u} \phi(x_i) \right)
$$

where:

- $x_i$ is the embedding of the $i$-th image,
- $\phi(\cdot)$ is a learnable transformation applied independently to each image,
- the aggregation operator (sum) is permutation-invariant,
- $\rho(\cdot)$ maps the aggregated representation to the final user embedding.

This formulation guarantees that the user representation does not depend on the order of images in the set.

Importantly, the user embedding is computed dynamically from image content and does not rely on stored user-specific parameters.


### 3.2 Deep Sets Block in BRIE

In practice, the Deep Sets block consists of two neural components:

- **$\phi$ network**: an MLP that transforms each image embedding independently.
- **$\rho$ network**: an MLP that refines the aggregated representation.

The aggregation is implemented using sum pooling:

$$
z_u = \sum_{i=1}^{N_u} \phi(x_i)
$$

$$
u = \rho(z_u)
$$

This design ensures:

- Permutation invariance  
- Variable-sized user histories  
- Independence from the number of users  

As a consequence, the number of model parameters does not grow with the number of users.


### 3.3 Ranking Objective (BPR)

The model preserves the Bayesian Pairwise Ranking (BPR) objective used in BRIE.

Given a triplet:

$$
(u, i^+, i^-)
$$

where:

- $u$ is the user embedding obtained via Deep Sets,
- $i^+$ is a positive image (uploaded by the user),
- $i^-$ is a negative sampled image,

the preference score is computed as:

$$
\hat{y}_{u,i} = u^\top v_i
$$

where $v_i$ is the latent embedding of image $i$.

The BPR loss is defined as:

$$
\mathcal{L}_{BPR} = - \log \sigma \left( \hat{y}_{u,i^+} - \hat{y}_{u,i^-} \right)
$$

This objective encourages the model to rank positive images higher than negative ones while maintaining the explainability-oriented structure of BRIE.


---

## 4. Experimental Setup

### 4.1 Environment

- Python ≥ 3.10  
- PyTorch and PyTorch Lightning  
- CUDA-enabled GPU recommended  

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 4.2 Datasets

Experiments were conducted on six real-world image-based recommendation datasets:

- **Gijón**
- **Barcelona**
- **Madrid**
- **New York**
- **Paris**
- **London**

Datasets follow the same structure as the original BRIE repository and should be placed under the `data/` directory.

---

## 5. Usage

### 5.1 Training

```bash
python main.py \
  --stage train \
  --city CITY_NAME \
  --model BRIE \
  --max_epochs EPOCHS \
  --batch_size BATCH_SIZE \
  --lr LR \
  --dropout DROPOUT \
  -d LATENT_DIM \
  --workers NUM_WORKERS
```

### Common training options

| Argument | Description |
|---------|-------------|
| `--city` | Dataset / city name (e.g. `barcelona`, `madrid`) |
| `--stage` | Execution stage: `train` or `test` |
| `--model` | Model name(s) to run (e.g. `BRIE`, `BRIE_DEEPSETS`) |
| `--batch_size` | Batch size |
| `--max_epochs` | Number of training epochs |
| `--lr` | Learning rate |
| `-d` | Latent embedding dimensionality |
| `--dropout` | Dropout rate |
| `--workers` | Number of dataloader workers |
| `--seed` | Random seed for reproducibility |

### Optional training flags

| Flag | Description |
|------|-------------|
| `--early_stopping` | Enable early stopping |
| `--no_validation` | Disable validation split |
| `--use_train_val` | Train using both train and validation splits |
| `--ckpt_path` | Path to a checkpoint to resume training |


### Deep Sets–specific options

| Flag | Description |
|------|-------------|
| `--max_user_images` | Maximum number of images per user set |


### 5.2 Evaluation

```bash
python main.py \
  --stage test \
  --city CITY_NAME \
  --model BRIE \
  --batch_size BATCH_SIZE \
  --workers NUM_WORKERS
```

### Listing all available options

To see the full and up-to-date list of command-line options and their default values, run:

```bash
python main.py --help
```
---

## 6. Results

Below we report test results across six cities.  
For each dataset, the best result is shown in **bold** and the second best in _italics_.


## Gijón

| Model | Params | MRecall@10 | MNDCG@10 | MAUC |
|-------|--------|------------|----------|------|
| RND | – | 0.373 | 0.185 | 0.487 |
| CNT | – | 0.464 | 0.218 | 0.546 |
| ELVis | 435,585 | 0.521 | 0.262 | 0.596 |
| MF-ELVis | 427,264 | 0.538 | 0.285 | 0.592 |
| **BRIE** | 427,264 | **0.607** | **0.333** | **0.643** |
| BRIE+DeepSets | 209,472 | _0.571_ | _0.303_ | _0.635_ |


## Barcelona

| Model | Params | MRecall@10 | MNDCG@10 | MAUC |
|-------|--------|------------|----------|------|
| RND | – | 0.409 | 0.186 | 0.502 |
| CNT | – | 0.443 | 0.219 | 0.554 |
| ELVis | 2,253,057 | 0.597 | 0.327 | 0.631 |
| MF-ELVis | 2,244,736 | 0.557 | 0.293 | 0.596 |
| **BRIE** | 2,244,736 | **0.630** | **0.368** | **0.663** |
| BRIE+DeepSets | 209,472 | _0.610_ | _0.343_ | _0.658_ |


## Madrid

| Model | Params | MRecall@10 | MNDCG@10 | MAUC |
|-------|--------|------------|----------|------|
| RND | – | 0.374 | 0.171 | 0.499 |
| CNT | – | 0.420 | 0.203 | 0.557 |
| ELVis | 2,948,609 | 0.572 | 0.314 | 0.638 |
| MF-ELVis | 2,940,288 | 0.528 | 0.279 | 0.601 |
| **BRIE** | 2,940,288 | **0.612** | **0.348** | **0.673** |
| BRIE+DeepSets | 209,472 | _0.597_ | _0.338_ | _0.668_ |


## New York

| Model | Params | MRecall@10 | MNDCG@10 | MAUC |
|-------|--------|------------|----------|------|
| RND | – | 0.374 | 0.168 | 0.502 |
| CNT | – | 0.431 | 0.217 | 0.563 |
| ELVis | 3,377,665 | 0.553 | 0.304 | 0.637 |
| MF-ELVis | 3,369,344 | 0.516 | 0.276 | 0.602 |
| **BRIE** | 3,369,344 | **0.598** | **0.341** | **0.677** |
| BRIE+DeepSets | 209,472 | _0.577_ | _0.328_ | _0.672_ |


## Paris

| Model | Params | MRecall@10 | MNDCG@10 | MAUC |
|-------|--------|------------|----------|------|
| RND | – | 0.459 | 0.209 | 0.502 |
| CNT | – | 0.499 | 0.245 | 0.557 |
| ELVis | 3,064,257 | 0.643 | 0.352 | 0.630 |
| MF-ELVis | 3,055,936 | 0.606 | 0.323 | 0.596 |
| **BRIE** | 3,055,936 | **0.669** | **0.391** | **0.666** |
| BRIE+DeepSets | 209,472 | _0.661_ | _0.375_ | _0.661_ |


## London

| Model | Params | MRecall@10 | MNDCG@10 | MAUC |
|-------|--------|------------|----------|------|
| RND | – | 0.342 | 0.155 | 0.500 |
| CNT | – | 0.400 | 0.200 | 0.562 |
| ELVis | 873,913 | 0.530 | 0.293 | 0.629 |
| MF-ELVis | 872,592 | 0.531 | 0.267 | 0.597 |
| **BRIE** | 872,592 | **0.563** | **0.318** | **0.665** |
| BRIE+DeepSets | 209,472 | _0.549_ | _0.312_ | _0.663_ |


## Observations

- **BRIE+DeepSets removes the dependency between model size and the number of users.**
- Parameter count becomes constant across datasets, enabling scalable deployment.
- Ranking performance remains consistently competitive across all cities.
- The performance gap with respect to BRIE is small, while achieving a substantially more scalable architecture.

---
## 7. Relationship to BRIE

This repository is an extension of the original BRIE model, developed in the context of an academic project.  
All credit for the original BRIE architecture, training strategy and evaluation protocol belongs to its authors.

Original repository:  
https://github.com/Kominaru/BRIE

---

## 8. License

This project follows the same license as the original BRIE repository.
