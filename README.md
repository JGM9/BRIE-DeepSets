# BRIE-DeepSets: Scalable User Representation with Deep Sets for Visual Explainability

**Authors (original BRIE):**  
Jorge Paz-Ruza*, Amparo Alonso-Betanzos  
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

Each user is represented as a set of visual embeddings extracted from images previously uploaded by that user. The representation follows the Deep Sets formulation:

\[
\mathbf{u} = \rho \left( \sum_{i=1}^{N_u} \phi(\mathbf{x}_i) \right)
\]

where:
- \(\mathbf{x}_i\) are image embeddings,
- \(\phi(\cdot)\) is a learnable transformation applied to each image,
- the aggregation is permutation-invariant,
- \(\rho(\cdot)\) produces the final user embedding.

This representation is computed dynamically and does not rely on stored user-specific parameters.

### 3.2 Ranking Objective

The model preserves the Bayesian Pairwise Ranking (BPR) loss used in BRIE, scoring triplets of the form:

\[
(u, i^+, i^-)
\]

where the user embedding \(u\) is obtained via Deep Sets, and \(i^+\), \(i^-\) correspond to positive and negative candidate images.

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

---

### Optional training flags

| Flag | Description |
|------|-------------|
| `--early_stopping` | Enable early stopping |
| `--no_validation` | Disable validation split |
| `--use_train_val` | Train using both train and validation splits |
| `--ckpt_path` | Path to a checkpoint to resume training |

---

### Deep Sets–specific options

| Flag | Description |
|------|-------------|
| `--max_user_images` | Maximum number of images per user set |

---

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

## 6. Results

### Gijón and Barcelona

| Model | MRecall@10 | MNDCG@10 | MAUC | | MRecall@10 | MNDCG@10 | MAUC |
|------|------------|----------|------|--|------------|----------|------|
| RND | 0.373 | 0.185 | 0.487 | | 0.409 | 0.186 | 0.502 |
| CNT | 0.464 | 0.218 | 0.546 | | 0.443 | 0.219 | 0.554 |
| ELVis | 0.521 | 0.262 | 0.596 | | 0.597 | 0.327 | 0.631 |
| MF-ELVis | 0.538 | 0.285 | 0.592 | | 0.557 | 0.293 | 0.596 |
| **BRIE** | **0.607** | **0.333** | **0.643** | | **0.630** | **0.368** | **0.663** |
| BRIE+DeepSets | 0.571 | 0.303 | 0.635 | | 0.610 | 0.343 | 0.658 |

---

### Madrid and New York

| Model | MRecall@10 | MNDCG@10 | MAUC | | MRecall@10 | MNDCG@10 | MAUC |
|------|------------|----------|------|--|------------|----------|------|
| RND | 0.374 | 0.171 | 0.499 | | 0.374 | 0.168 | 0.502 |
| CNT | 0.420 | 0.203 | 0.557 | | 0.431 | 0.217 | 0.563 |
| ELVis | 0.572 | 0.314 | 0.638 | | 0.553 | 0.304 | 0.637 |
| MF-ELVis | 0.528 | 0.279 | 0.601 | | 0.516 | 0.276 | 0.602 |
| **BRIE** | **0.612** | **0.348** | **0.673** | | **0.598** | **0.341** | **0.677** |
| BRIE+DeepSets | 0.597 | 0.338 | 0.668 | | 0.577 | 0.328 | 0.672 |

---

### Paris and London

| Model | MRecall@10 | MNDCG@10 | MAUC | | MRecall@10 | MNDCG@10 | MAUC |
|------|------------|----------|------|--|------------|----------|------|
| RND | 0.459 | 0.209 | 0.502 | | 0.342 | 0.155 | 0.500 |
| CNT | 0.499 | 0.245 | 0.557 | | 0.400 | 0.200 | 0.562 |
| ELVis | 0.643 | 0.352 | 0.630 | | 0.530 | 0.293 | 0.629 |
| MF-ELVis | 0.606 | 0.323 | 0.596 | | 0.531 | 0.267 | 0.597 |
| **BRIE** | **0.669** | **0.391** | **0.666** | | **0.563** | **0.318** | **0.665** |
| BRIE+DeepSets | 0.661 | 0.375 | 0.661 | | 0.549 | 0.312 | 0.663 |

---

## 7. Relationship to BRIE

This repository is an extension of the original BRIE model, developed in the context of an academic project.  
All credit for the original BRIE architecture, training strategy and evaluation protocol belongs to its authors.

Original repository:  
https://github.com/Kominaru/BRIE

---

## 8. License

This project follows the same license as the original BRIE repository.
