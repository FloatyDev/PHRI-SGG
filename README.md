<div align="center">

# PHRI-SGG: Probabilistic Hierarchical Relation Inference for Scene Graph Generation

**Thesis Project — Giorgos Sygkelakis, January 2026**

> Built on top of [EGTR](https://github.com/naver-ai/egtr) (CVPR 2024).  
> This repository extends EGTR with a probabilistic hierarchical relation inference framework to mitigate long-tail bias in Scene Graph Generation.

</div>

---

## Abstract

Scene Graph Generation (SGG) is dominated by long-tailed predicate distributions, causing flat classifiers to over-predict generic relations like *on* and *wearing*, at the expense of informative, fine-grained semantics. To address this problem, we present **PHRI-SGG** (Probabilistic Hierarchical Relation Inference for Scene Graph Generation), where a super-classifier first predicts a coarse semantic family and statically routes features to specialized family experts that refine the predicate. We realize this with:

- **(a)** Logit-space fusion for soft routing
- **(b)** A separate connectivity gate to decouple edge existence from semantics
- **(c)** Weight transfer from a converged flat baseline to warm-start the experts
- **(d)** Router pretraining via family-level distillation
- **(e)** A supervised contrastive objective, restricted within families

PHRI-SGG improves mean recall on VG150 while overall R@20/50/100 decreases modestly, reflecting a deliberate trade-off from generic to specific predictions. Gains concentrate on tail predicates and coincide with fewer hallucinations of head classes, demonstrating that explicit hierarchical decomposition yields more semantically precise scene graphs under long-tail conditions.

---

## Architecture

The PHRI-SGG pipeline builds on EGTR's efficient one-stage graph extraction and replaces its flat 50-class relation head with a modular, two-step hierarchical inference system.

### Overview

1. **Feature Extraction (EGTR backbone):** A ResNet-50 + Deformable DETR encoder-decoder extracts per-object query features. Relation edge features `z_sub,obj` are constructed from attention by-products (Q/K projections across decoder layers) aggregated via a learnable gated sum.

2. **Connectivity Gate:** A binary classification head predicts whether a relation *exists* between a pair, decoupling the *existence* decision from *semantic* classification.

3. **Super-Relation Router:** A lightweight classifier predicts one of three coarse semantic families  (**Geometric**, **Possessive**, or **Semantic**) acting as a soft router over the shared edge features.

4. **Family Experts:** Three specialized sub-networks, each trained exclusively on their own relation family:
   - **Geometric Expert** — spatial relations (*on, under, near, behind*, ...)
   - **Possessive Expert** — ownership/part-whole relations (*has, part of, wearing*, ...)
   - **Semantic Expert** — active interactions (*riding, eating, looking at*, ...)

5. **Logit-Space Fusion:** The expert logits are fused back into a global N×N×50 prediction matrix and normalized via Softmax before evaluation.

### Relation Family Distribution in VG150

| Family      | Count   | Percentage |
|-------------|---------|------------|
| Geometric   | 157,252 | 49.82%     |
| Possessive  | 130,928 | 41.48%     |
| Semantic    | 27,462  | 8.70%      |
| **Total**   | **315,642** | **100%** |

The Semantic family — containing the most informative interactions — represents fewer than 9% of training samples, making it the primary target of PHRI-SGG's debiasing effort.

---

## Key Contributions

- **Hierarchical SGG framework:** Replaces the standard flat classifier with a statically-routed modular architecture combining coarse family routing with fine-grained predicate prediction.

- **Long-tail debiasing by design:** By isolating the Semantic expert from geometric gradient dominance, PHRI-SGG enables learning of subtle visual features for rare classes like *parked on*, *attached to*, and *standing on* that had near-zero recall in the flat baseline.

- **Hallucination reduction:** The hierarchical constraint forces the model to be more discriminative, reducing false positives for dominant head classes like *wearing* and *on*.

- **Specificity over generality:** Generic predictions (*on*, *at*) are redirected to their precise counterparts (*sitting on*, *riding*, *parked on*), producing semantically richer scene graphs.

---

## Results on VG150 (SGDet, graph-constraint)

| Metric   | EGTR (baseline) | PHRI-SGG | Abs. Change | % Change |
|----------|:-----------:|:--------:|:-----------:|:--------:|
| mR@20    | 0.055       | **0.071** | +0.016     | **+29.1%** |
| mR@50    | 0.079       | **0.098** | +0.019     | **+24.1%** |
| mR@100   | 0.101       | **0.119** | +0.018     | **+17.8%** |
| R@20     | 0.235       | 0.215    | -0.020      | -8.5%    |
| R@50     | 0.302       | 0.277    | -0.025      | -8.2%    |
| R@100    | 0.344       | 0.318    | -0.026      | -7.5%    |

The modest drop in R@K is a **feature, not a bug** — it reflects the model redirecting probability mass away from generic head classes toward semantically informative tail predicates.

### Selected Per-Class Gains (R@100)

| Predicate    | Family   | EGTR  | PHRI-SGG | Impact     |
|--------------|----------|-------|----------|------------|
| parked on    | Semantic | 0.017 | 0.445    | +2516.2%   |
| attached to  | Semantic | 0.006 | 0.056    | +831.7%    |
| standing on  | Semantic | 0.025 | 0.183    | +631.6%    |
| walking on   | Semantic | 0.050 | 0.241    | +381.5%    |
| riding       | Semantic | 0.277 | 0.491    | +77.1%     |
| sitting on   | Semantic | 0.132 | 0.248    | +87.9%     |

---

## Loss Function

The total training objective is a weighted sum of four components:

```
L_total = λ_rel (L_super + L_experts) + λ_conn L_conn + λ_supcon L_supcon + L_od
```

| Component      | Description                                              | Coefficient |
|----------------|----------------------------------------------------------|-------------|
| `L_super`      | Cross-entropy on the coarse family prediction            | λ_rel = 1   |
| `L_experts`    | Conditional masked cross-entropy within each family      | λ_rel = 1   |
| `L_conn`       | Binary connectivity (foreground vs. background pairs)    | λ_conn = 30 |
| `L_supcon`     | Supervised contrastive loss on shared embeddings         | λ_supcon = 1|
| `L_od`         | Standard DETR object detection loss                      | λ_cls=1, λ_L1=5, λ_IoU=2 |

---

## Installation

### Dependencies

Same environment as EGTR:

```bash
# Docker image: nvcr.io/nvidia/pytorch:21.11-py3
pip install -r requirements.txt
cd lib/fpn
sh make.sh
```

### Dataset

Evaluated on **Visual Genome (VG150)** using the standard splits from [Neural Motifs](https://github.com/rowanz/neural-motifs).

- Download instructions: https://github.com/yrcong/RelTR/blob/main/data/README.md

```
dataset/
└── visual_genome/
    └── images/
        train.json
        val.json
        test.json
        rel.json
```

---

## Training

Both stages use `train_phri.py` with a YAML config file. Pass `--config config_train.yaml`
instead of individual flags. The script saves a copy of the config to the training log
directory automatically. CLI flags always override the config file if provided alongside `--config`.

### Stage 1 — Pretrain the Super-Relation Router

> Branch: `super_classifier_training`

The super-classifier is pretrained in isolation using a `DualHeadRelationClassifier`.
This module shares the frozen MLP layers of a converged Flat-50 checkpoint and adds
a lightweight 3-class head on top (Geometric / Possessive / Semantic).
Only the `super_head` linear layer is trained.
```bash
python train_phri.py --config config_train.yaml
```

### Stage 2 — Train PHRI-SGG with Expert Modules

> Branch: `main`

The pretrained super-classifier checkpoint from Stage 1 warm-starts the full
PHRI-SGG model with three family experts. The ResNet-50 backbone and Deformable
DETR encoder-decoder are frozen. Only the hierarchical heads and final projection
layers receive gradient updates.
```bash
python train_phri.py --config config_train.yaml
```

### Config file (`config_train.yaml`)

A `config_train.yaml` is provided at the root of the repository. Key fields to update
before running:
```yaml
# --- Paths (update these) ---
data_path: "dataset/visual_genome"
output_path: "results/"
pretrained: "path/to/pretrained_detr_checkpoint"
flat_path: "artifacts/"          # path to converged Flat-50 checkpoint (Stage 2)

# --- Hierarchical settings ---
hierarchical: true
num_geometric: 15
num_possessive: 11
num_semantic: 24
super_weight: 1
train_head: true

# --- Loss coefficients ---
rel_loss_coefficient: 15.0
connectivity_loss_coefficient: 30.0

# --- Training ---
gpus: 1
batch_size: 4
max_epochs: 20
lr: 2.0e-6
lr_initialized: 2.0e-4
lr_backbone: 2.0e-7
```

All available options and their defaults are documented inside `config_train.yaml`.

**Hardware:** 2× NVIDIA GPU, batch size 16 per GPU, AdamW optimizer,
learning rate 2×10⁻⁵ for hierarchical heads and 2×10⁻⁶ for unfrozen decoder components.
---

## Repository Structure

```
PHRI-SGG/
├── model/
│   ├── egtr.py              # Core model — includes hierarchical heads
│   └── deformable_detr.py   # Backbone config
├── lib/
│   └── evaluation/
│       └── sg_eval.py       # SGG evaluation (R@K, mR@K)
├── train_phri.py            # PHRI-SGG hierarchical training / Super-Classifier training (super_classifier_training branch)
├── pretrain_detr.py         # Object detector pretraining
└── requirements.txt
```

---

## Citation

This work builds on EGTR:

```bibtex
@inproceedings{im2024egtr,
  title     = {EGTR: Extracting Graph from Transformer for Scene Graph Generation},
  author    = {Im, Jinbae and others},
  booktitle = {CVPR},
  year      = {2024}
}
```

---

## Acknowledgements

The EGTR codebase provided the foundational architecture for feature extraction, connectivity prediction, and relation smoothing on which PHRI-SGG is built.
I also want to thank my supervisor Prof. Antonis Argyros and Assistant Professor Konstantinos Papoutsakis for their collaboration and their help in the development of this project.
Finally, the computational resources for this work were provided by the Human-Centered Computer Vision group (HCCV) of the Computational Vision and Robotics Laboratory (CVRL) withing the Institute of Computer Science, Foundation of Research and Technology-Hellas (FORTH).
