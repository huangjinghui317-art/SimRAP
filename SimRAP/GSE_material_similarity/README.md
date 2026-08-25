# GSE material similarity

## Module mapping

- `constants.py`: original global constants.
- `utils.py`: original utility / logging / resource-monitor blocks.
- `chemistry.py`: original periodic-table helper blocks.
- `descriptor.py`: original structure descriptor block.
- `tree.py`: original trie / multi-way tree block.
- `similarity.py`: original L1-L4 similarity block.
- `build_index.py`: original Step 1 block.
- `retrieval_core.py`: original retrieval-helper block.
- `query_io.py`: original CSV/checkpoint/output helper blocks.
- `retrieve_step.py`: original Step 2 block.
- `cli.py`: original CLI block.
- `gse_similarity_tree.py`: total entry script.

## Usage

The top-level command remains the same.

### Step 1

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

### Step 2

```bash
python SimRAP/GSE_material_similarity/gse_similarity_tree.py \
  --step 2 \
  --train-csv data/jarvis2021/targets/bulk_modulus_kv/train.csv \
  --query-csvs data/jarvis2021/targets/bulk_modulus_kv/train.csv \
  --output-dir data/gse_similarity_results/bulk_modulus_kv \
  --index-file data/similarity_index/bulk_modulus_kv/similarity_index.pkl \
  --k 5
```

### Build + retrieve

```bash
python material_similarity_tree_knn.py \
  --step all \
  --train-csv /path/to/train.csv \
  --query-csvs /path/to/val.csv \
  --output-dir output \
  --k 5
```
