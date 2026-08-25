# SimRAP Reproduction Guide

This document provides a reproduction guide for **SimRAP**, covering environment setup, JARVIS2021 data preparation, GSE similarity retrieval, and wrapper training/evaluation.

---

## 1. Environment Setup

We recommend using an isolated Conda environment.

```bash
conda create -n simrap python=3.13
conda activate simrap
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```



---

## 2. Data Preparation

### 2.1 Prepare the JARVIS2021 ALIGNN splits

First, prepare the JARVIS2021 ALIGNN data splits:

```bash
python data/prepare_jarvis2021_alignn_splits.py \
    --target all \
    --output-dir data/jarvis2021
```

The supported targets include:

```text
bulk_modulus_kv
shear_modulus_gv
mbj_bandgap
optb88vdw_bandgap
magmom_oszicar
...
```

---

### 2.2 Add complete CIF structures

The generated split files should then be supplemented with complete CIF structures.

For example, for `bulk_modulus_kv`:

```bash
python data/add_cif_to_alignn_splits.py \
  --split-dir data/jarvis2021/targets/bulk_modulus_kv
```

For other properties, replace `bulk_modulus_kv` with one of:

```text
shear_modulus_gv
mbj_bandgap
optb88vdw_bandgap
magmom_oszicar
```

After this step, the training and validation sets can be found at:

```text
data/jarvis2021/targets/bulk_modulus_kv/train.csv
data/jarvis2021/targets/bulk_modulus_kv/val.csv
```

The same directory structure is used for the other properties.

---

### 2.3 Prepare `backbone_output`

SimRAP uses the prediction of an existing material-property model as an additional prior. Therefore, the dataset used by the wrapper should contain a column named:

```text
backbone_output
```

You may run your own backbone model on the prepared structures and write its prediction into this column.

Alternatively, for direct reproduction, we provide processed training and validation sets with precomputed ALIGNN backbone outputs under:

```text
resources/dataset_with_ALIGNN/
```

For example:

```text
resources/dataset_with_ALIGNN/bulk_modulus/train.csv
resources/dataset_with_ALIGNN/bulk_modulus/val.csv
```

#### Optional distribution adjustment

If you generate `backbone_output` using your own backbone model, it may be useful to perturb or adjust the training-set backbone outputs.

The corresponding utility is:

```text
data/match_backbone_output_distribution.py
```

For detailed usage of this script, please refer to the comments at the top of this py file.


---

## 3. GSE Similarity Retrieval

The GSE retrieval procedure contains two stages:

1. build the hierarchical similarity index;
2. query the index to retrieve the top-k structural neighbors.

The example below uses `bulk_modulus_kv`.

---

### 3.1 Stage 1: Build the GSE similarity index

```bash
python SimRAP/GSE_material_similarity/gse_similarity_tree.py \
  --step 1 \
  --train-csv data/jarvis2021/targets/bulk_modulus_kv/train.csv \
  --output-dir data/similarity_index/bulk_modulus_kv \
  --cif-column cif \
  --target-column target_value \
  --symprec 0.01 \
  --angle-tolerance 5 \
  --standardize refined
```

The generated index is stored under:

```text
data/similarity_index/bulk_modulus_kv/
```

The main index file used in the next stage is:

```text
data/similarity_index/bulk_modulus_kv/similarity_index.pkl
```

---

### 3.2 Stage 2: Retrieve structural neighbors

```bash
python SimRAP/GSE_material_similarity/gse_similarity_tree.py \
  --step 2 \
  --train-csv data/jarvis2021/targets/bulk_modulus_kv/train.csv \
  --query-csvs data/jarvis2021/targets/bulk_modulus_kv/train.csv \
  --output-dir data/gse_similarity_results/bulk_modulus_kv \
  --index-file data/similarity_index/bulk_modulus_kv/similarity_index.pkl \
  --k 5
```

This retrieves up to five structurally similar materials for each query structure.

The retrieval results are written under:

```text
data/gse_similarity_results/bulk_modulus_kv/
```

For another property, use the corresponding target directory in both stages.

If retrieval results are also required for the validation set, use the same index built from the training set and query the corresponding validation CSV.

---

## 4. Training and Evaluation

For the fastest reproduction, you can directly use the datasets with ALIGNN backbone outputs provided in:

```text
resources/dataset_with_ALIGNN/
```

The following example trains SimRAP for **bulk modulus**.

```bash
python SimRAP/wrapper/wrapper.py \
    --train-csv resources/dataset_with_ALIGNN/bulk_modulus/train.csv \
    --val-csv resources/dataset_with_ALIGNN/bulk_modulus/val.csv \
    --output-dir results/train/bulk_modulus \
    --backbone_output-col backbone_output \
    --epochs 300 \
    --batch-size 32 \
    --val-every 1 \
    --save-every 1 \
    --max-ckpts 300
```

During training, validation is performed every `--val-every` epochs and checkpoints are saved every `--save-every` epochs.

The main outputs are stored in:

```text
results/train/bulk_modulus/
```

Typical outputs include:

```text
run_config.json
history.csv
training_curves.png
checkpoints/
```

In our experiments, a strong validation result is usually obtained after approximately **100–250 epochs**, corresponding to roughly **10–20 hours of training** depending on the hardware and runtime configuration.

If sufficient memory is available, increasing:

```text
--batch-size
```

can substantially accelerate training.

Because validation is performed periodically during training, the checkpoint corresponding to the best validation result can be selected directly from the saved checkpoints.

---

## 5. Reproducing Other Properties

The overall reproduction pipeline is identical for the other supported properties:

```text
JARVIS2021 ALIGNN split
        |
        v
Add complete CIF structures
        |
        v
Generate backbone_output
        |
        v
Build GSE similarity index
        |
        v
Retrieve structural neighbors
        |
        v
Train and evaluate SimRAP
```

Available JARVIS2021 targets include:

```text
bulk_modulus_kv
shear_modulus_gv
mbj_bandgap
optb88vdw_bandgap
magmom_oszicar
```

The wrapper itself is designed to use a generic:

```text
backbone_output
```

and therefore does not require a specific backbone architecture.

---
