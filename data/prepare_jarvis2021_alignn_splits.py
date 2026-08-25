#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_jarvis2021_alignn_splits.py

Download the JARVIS-DFT 2021 dataset and generate train / val / test splits
following the ALIGNN 2021 data-splitting logic.

Core reproduction logic
------------
1. Dataset:
       jarvis.db.figshare.data("dft_3d_2021")

   Corresponding historical snapshot:
       jdft_3d-8-18-2021.json

2. Filter each target independently:
       None
       "na"
       NaN

3. ALIGNN default split:
       random_seed = 123
       train_ratio = 0.8
       val_ratio = 0.1
       test_ratio = 0.1

4. Original ALIGNN shuffle:
       random.seed(123)
       random.shuffle(ids)

5. Original ALIGNN split:
       id_train = ids[:n_train]

       id_val = ids[
           -(n_val + n_test):-n_test
       ]

       id_test = ids[-n_test:]

Note:
    Because ALIGNN uses
        int(0.8*N)
        int(0.1*N)
        int(0.1*N)

    For some values of N, the sum of these three subsets may be 1--2 samples smaller than N.
    This script does not silently assign such samples to any split.
    Instead, they are saved separately as unused,
    in order to reproduce the original logic as faithfully as possible.

Outputs:
    output_dir/
    ├── raw/
    │   └── jarvis_dft_3d_2021.json.gz
    │
    ├── split_summary.csv
    │
    └── targets/
        └── bulk_modulus_kv/
            ├── metadata.json
            ├── ids_train_val_test.json
            ├── all_valid_with_split.csv
            ├── train.csv
            ├── val.csv
            ├── test.csv
            ├── unused.csv
            │
            └── records/              # only when --save-records is enabled
                ├── train.json.gz
                ├── val.json.gz
                ├── test.json.gz
                └── unused.json.gz

Usage examples
--------

1. Process all 29 ALIGNN 2021 regression targets at once (about 30 seconds):

python data/prepare_jarvis2021_alignn_splits.py \
    --target all \
    --output-dir data/jarvis2021


2. List the 29 supported ALIGNN regression targets:

python data/prepare_jarvis2021_alignn_splits.py \
    --list-targets

ALIGNN JARVIS-DFT 2021 regression targets:

 1. formation_energy_peratom       Formation energy
 2. optb88vdw_bandgap              Bandgap (OPT)
 3. optb88vdw_total_energy         Total energy
 4. ehull                          Ehull
 5. mbj_bandgap                    Bandgap (MBJ)
 6. bulk_modulus_kv                Kv / Bulk modulus
 7. shear_modulus_gv               Gv / Shear modulus
 8. magmom_oszicar                 Magnetic moment
 9. slme                           SLME
10. spillage                       Spillage
11. kpoint_length_unit             K-point length
12. encut                          Plane-wave cutoff
13. epsx                           epsilon_x (OPT)
14. epsy                           epsilon_y (OPT)
15. epsz                           epsilon_z (OPT)
16. mepsx                          epsilon_x (MBJ)
17. mepsy                          epsilon_y (MBJ)
18. mepsz                          epsilon_z (MBJ)
19. dfpt_piezo_max_dielectric      DFPT dielectric
20. dfpt_piezo_max_dij             Maximum piezoelectric strain coefficient
21. dfpt_piezo_max_eij             Maximum piezoelectric stress coefficient
22. exfoliation_energy             Exfoliation energy
23. max_efg                        Maximum EFG
24. avg_elec_mass                  Average electron effective mass
25. avg_hole_mass                  Average hole effective mass
26. n-Seebeck                      n-type Seebeck
27. n-powerfact                    n-type power factor
28. p-Seebeck                      p-type Seebeck
29. p-powerfact                    p-type power factor

Total: 29
"""

import argparse
import csv
import gzip
import json
import math
import random
import sys
from pathlib import Path

import numpy as np


# ============================================================
# ALIGNN JARVIS-DFT 2021 regression targets
#
# Field names are the actual target names used by ALIGNN / JARVIS
# Human-readable names are used for display in the output.
# ============================================================

ALIGNN_REGRESSION_TARGETS = {
    # --------------------------------------------------------
    # Energetics / electronic
    # --------------------------------------------------------
    "formation_energy_peratom": "Formation energy",
    "optb88vdw_bandgap": "Bandgap (OPT)",
    "optb88vdw_total_energy": "Total energy",
    "ehull": "Ehull",
    "mbj_bandgap": "Bandgap (MBJ)",

    # --------------------------------------------------------
    # Mechanical
    # --------------------------------------------------------
    "bulk_modulus_kv": "Kv / Bulk modulus",
    "shear_modulus_gv": "Gv / Shear modulus",

    # --------------------------------------------------------
    # Magnetic / photovoltaic / topology
    # --------------------------------------------------------
    "magmom_oszicar": "Magnetic moment",
    "slme": "SLME",
    "spillage": "Spillage",

    # --------------------------------------------------------
    # Calculation parameters
    # --------------------------------------------------------
    "kpoint_length_unit": "K-point length",
    "encut": "Plane-wave cutoff",

    # --------------------------------------------------------
    # OPT dielectric tensor
    # --------------------------------------------------------
    "epsx": "epsilon_x (OPT)",
    "epsy": "epsilon_y (OPT)",
    "epsz": "epsilon_z (OPT)",

    # --------------------------------------------------------
    # MBJ dielectric tensor
    # --------------------------------------------------------
    "mepsx": "epsilon_x (MBJ)",
    "mepsy": "epsilon_y (MBJ)",
    "mepsz": "epsilon_z (MBJ)",

    # --------------------------------------------------------
    # DFPT / piezoelectric
    # --------------------------------------------------------
    "dfpt_piezo_max_dielectric": "DFPT dielectric",
    "dfpt_piezo_max_dij": "Maximum piezoelectric strain coefficient",
    "dfpt_piezo_max_eij": "Maximum piezoelectric stress coefficient",

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------
    "exfoliation_energy": "Exfoliation energy",
    "max_efg": "Maximum EFG",

    # --------------------------------------------------------
    # Effective mass
    # --------------------------------------------------------
    "avg_elec_mass": "Average electron effective mass",
    "avg_hole_mass": "Average hole effective mass",

    # --------------------------------------------------------
    # Thermoelectric
    # --------------------------------------------------------
    "n-Seebeck": "n-type Seebeck",
    "n-powerfact": "n-type power factor",
    "p-Seebeck": "p-type Seebeck",
    "p-powerfact": "p-type power factor",
}


# ============================================================
# JSON helpers
# ============================================================

def json_default(obj):
    """
    Allow json.dump to handle common NumPy types.
    """

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(
        f"Object of type {type(obj).__name__} "
        f"is not JSON serializable."
    )


def save_json(path, obj, indent=2):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=indent,
            default=json_default,
        )


def save_json_gz(path, obj):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with gzip.open(
        path,
        "wt",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            ensure_ascii=False,
            default=json_default,
        )


# ============================================================
# Validate target values
# ============================================================

def is_valid_alignn_target(value):
    """
    Determine whether a target value is valid, following the original ALIGNN logic as closely as possible.

    The original ALIGNN preprocessing mainly checks:
        value is not None
        value != "na"
        not math.isnan(value)
    """

    if value is None:
        return False

    if isinstance(value, str):

        if value.strip().lower() == "na":
            return False

        # Standard JARVIS regression targets should be numeric.
        # Numeric strings are accepted for improved compatibility.
        try:
            value = float(value)
        except ValueError:
            return False

    try:

        if math.isnan(value):
            return False

    except TypeError:

        return False

    return True


# ============================================================
# Load JARVIS-DFT 2021
# ============================================================

def load_jarvis_2021():

    try:
        from jarvis.db.figshare import data
    except ImportError as exc:

        raise RuntimeError(
            "\njarvis-tools was not detected.\n\n"
            "Please install it first:\n\n"
            "    pip install -U jarvis-tools\n"
        ) from exc

    print()
    print("=" * 90)
    print("Downloading / loading JARVIS-DFT 2021")
    print("=" * 90)

    try:

        dataset = data("dft_3d_2021")

    except Exception as exc:

        raise RuntimeError(
            "\nFailed to load data('dft_3d_2021').\n\n"
            "Please update jarvis-tools first:\n\n"
            "    pip install -U jarvis-tools\n\n"
            f"Original error: {repr(exc)}"
        ) from exc

    print()
    print(f"Raw materials: {len(dataset):,}")

    if len(dataset) == 0:
        raise RuntimeError(
            "The JARVIS-DFT 2021 dataset is empty."
        )

    # Basic checks
    if "jid" not in dataset[0]:
        raise RuntimeError(
            "The jid field was not found in the dataset."
        )

    return dataset


# ============================================================
# Original ALIGNN split
# ============================================================

def alignn_split_indices(
    total_size,
    seed=123,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
):
    """
    Reproduce the original ALIGNN 2021 data.py implementation as closely as possible.

    Original logic:

        n_train = int(train_ratio * total_size)
        n_test  = int(test_ratio  * total_size)
        n_val   = int(val_ratio   * total_size)

        ids = list(np.arange(total_size))

        random.seed(split_seed)
        random.shuffle(ids)

        id_train = ids[:n_train]

        id_val = ids[
            -(n_val + n_test):-n_test
        ]

        id_test = ids[-n_test:]

    """

    n_train = int(
        train_ratio * total_size
    )

    n_val = int(
        val_ratio * total_size
    )

    n_test = int(
        test_ratio * total_size
    )

    ids = list(
        np.arange(total_size)
    )

    # ========================================================
    # Important:
    # Use Python random rather than np.random
    # ========================================================

    random.seed(seed)

    random.shuffle(ids)

    # ========================================================
    # Follow the ALIGNN slicing logic exactly
    # ========================================================

    id_train = ids[:n_train]

    if n_test > 0:

        id_val = ids[
            -(n_val + n_test):-n_test
        ]

        id_test = ids[-n_test:]

    else:

        id_val = ids[-n_val:] if n_val > 0 else []

        id_test = []

    # ========================================================
    # int() truncation may leave unused samples
    # ========================================================

    used = (
        set(id_train)
        | set(id_val)
        | set(id_test)
    )

    id_unused = [
        idx
        for idx in ids
        if idx not in used
    ]

    # numpy int -> Python int
    id_train = [
        int(x)
        for x in id_train
    ]

    id_val = [
        int(x)
        for x in id_val
    ]

    id_test = [
        int(x)
        for x in id_test
    ]

    id_unused = [
        int(x)
        for x in id_unused
    ]

    return {
        "train": id_train,
        "val": id_val,
        "test": id_test,
        "unused": id_unused,
    }


# ============================================================
# CSV
# ============================================================

def save_split_csv(
    path,
    records,
    target,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "split",
        "filtered_index",
        "original_index",
        "jid",
        "target_name",
        "target_value",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for rec in records:

            writer.writerow({
                "split": rec["split"],
                "filtered_index": rec["filtered_index"],
                "original_index": rec["original_index"],
                "jid": rec["jid"],
                "target_name": target,
                "target_value": rec["target_value"],
            })


# ============================================================
# Single target
# ============================================================

def process_target(
    dataset,
    target,
    output_root,
    seed=123,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    save_records=False,
):

    display_name = ALIGNN_REGRESSION_TARGETS.get(
        target,
        target,
    )

    print()
    print("=" * 90)
    print(f"Target: {target}")
    print(f"Property: {display_name}")
    print("=" * 90)

    # ========================================================
    # 1. Filter the target first
    #
    # Preserve the original JARVIS data order here.
    # ========================================================

    filtered = []

    for original_index, item in enumerate(dataset):

        if target not in item:
            continue

        value = item[target]

        if not is_valid_alignn_target(value):
            continue

        filtered.append({
            "original_index": original_index,
            "item": item,
        })

    n_total = len(dataset)
    n_valid = len(filtered)

    print(f"Raw dataset : {n_total:,}")
    print(f"Valid target: {n_valid:,}")
    print(
        f"Filtered out: "
        f"{n_total - n_valid:,}"
    )

    if n_valid == 0:

        print(
            f"[WARNING] target={target} "
            "No valid samples; skipping."
        )

        return None

    # ========================================================
    # 2. ALIGNN split
    # ========================================================

    split_indices = alignn_split_indices(
        total_size=n_valid,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    # ========================================================
    # 3. Build split records
    # ========================================================

    split_records = {
        "train": [],
        "val": [],
        "test": [],
        "unused": [],
    }

    for split_name, indices in split_indices.items():

        for filtered_index in indices:

            entry = filtered[
                filtered_index
            ]

            item = entry["item"]

            split_records[
                split_name
            ].append({
                "split": split_name,
                "filtered_index": filtered_index,
                "original_index": entry["original_index"],
                "jid": item["jid"],
                "target_value": item[target],
                "item": item,
            })

    n_train = len(
        split_records["train"]
    )

    n_val = len(
        split_records["val"]
    )

    n_test = len(
        split_records["test"]
    )

    n_unused = len(
        split_records["unused"]
    )

    print()
    print("ALIGNN split:")
    print(
        f"  train  : {n_train:,}"
    )
    print(
        f"  val    : {n_val:,}"
    )
    print(
        f"  test   : {n_test:,}"
    )
    print(
        f"  unused : {n_unused:,}"
    )

    print(
        f"  total  : "
        f"{n_train + n_val + n_test + n_unused:,}"
    )

    # ========================================================
    # 4. Output directory
    # ========================================================

    target_dir = (
        Path(output_root)
        / "targets"
        / target
    )

    target_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 5. Save metadata
    # ========================================================

    metadata = {
        "dataset": "dft_3d_2021",
        "historical_file": "jdft_3d-8-18-2021.json",
        "target": target,
        "property_name": display_name,

        "raw_samples": n_total,
        "valid_samples": n_valid,

        "random_seed": seed,

        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,

        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_unused": n_unused,

        "shuffle": "python random.shuffle",
        "split_style": "ALIGNN 2021",

        "filter_before_split": True,

        "invalid_values": [
            "None",
            "na",
            "NaN",
        ],
    }

    save_json(
        target_dir / "metadata.json",
        metadata,
    )

    # ========================================================
    # 6. Save ALIGNN-style JID splits
    # ========================================================

    ids_json = {
        "id_train": [
            x["jid"]
            for x
            in split_records["train"]
        ],

        "id_val": [
            x["jid"]
            for x
            in split_records["val"]
        ],

        "id_test": [
            x["jid"]
            for x
            in split_records["test"]
        ],

        # The original ALIGNN file does not contain this field;
        # it is saved here for auditing int() truncation.
        "id_unused": [
            x["jid"]
            for x
            in split_records["unused"]
        ],
    }

    save_json(
        target_dir
        / "ids_train_val_test.json",
        ids_json,
    )

    # ========================================================
    # 7. Save CSV files
    # ========================================================

    for split_name in [
        "train",
        "val",
        "test",
        "unused",
    ]:

        save_split_csv(
            target_dir
            / f"{split_name}.csv",
            split_records[split_name],
            target,
        )

    # All valid samples + split labels
    all_records = (
        split_records["train"]
        + split_records["val"]
        + split_records["test"]
        + split_records["unused"]
    )

    # Restore the original post-filter JARVIS order using filtered_index
    all_records = sorted(
        all_records,
        key=lambda x: x[
            "filtered_index"
        ],
    )

    save_split_csv(
        target_dir
        / "all_valid_with_split.csv",
        all_records,
        target,
    )

    # ========================================================
    # 8. Optional: save complete JARVIS records
    #
    # Each item contains atoms and can therefore be used directly to reconstruct the structure.
    # ========================================================

    if save_records:

        records_dir = (
            target_dir
            / "records"
        )

        records_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for split_name in [
            "train",
            "val",
            "test",
            "unused",
        ]:

            full_records = [
                rec["item"]
                for rec
                in split_records[
                    split_name
                ]
            ]

            save_json_gz(
                records_dir
                / f"{split_name}.json.gz",
                full_records,
            )

    return metadata


# ============================================================
# summary CSV
# ============================================================

def save_summary(
    path,
    summaries,
):

    fields = [
        "target",
        "property_name",
        "raw_samples",
        "valid_samples",
        "n_train",
        "n_val",
        "n_test",
        "n_unused",
        "random_seed",
        "train_ratio",
        "val_ratio",
        "test_ratio",
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for item in summaries:

            writer.writerow({
                key: item.get(key)
                for key in fields
            })


# ============================================================
# Print targets
# ============================================================

def print_targets():

    print()
    print(
        "ALIGNN JARVIS-DFT 2021 "
        "regression targets:"
    )
    print()

    for i, (
        target,
        name,
    ) in enumerate(
        ALIGNN_REGRESSION_TARGETS.items(),
        start=1,
    ):

        print(
            f"{i:2d}. "
            f"{target:30s} "
            f"{name}"
        )

    print()
    print(
        f"Total: "
        f"{len(ALIGNN_REGRESSION_TARGETS)}"
    )


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Download JARVIS-DFT 2021 and "
            "reproduce ALIGNN train/val/test splits."
        )
    )

    parser.add_argument(
        "--target",
        type=str,
        default="bulk_modulus_kv",
        help=(
            "Regression target. "
            "For example, bulk_modulus_kv, "
            "optb88vdw_bandgap, "
            "or all."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="jarvis2021_alignn_splits",
        help="Output directory",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help=(
            "ALIGNN split random seed. "
            "Default: 123."
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--save-records",
        action="store_true",
        help=(
            "Additionally save complete JARVIS "
            "records for each split, including atoms."
        ),
    )

    parser.add_argument(
        "--no-save-raw",
        action="store_true",
        help=(
            "Do not additionally save the complete "
            "JARVIS-DFT 2021 json.gz file."
        ),
    )

    parser.add_argument(
        "--list-targets",
        action="store_true",
        help="List the 29 ALIGNN 2021 regression targets.",
    )

    args = parser.parse_args()

    # ========================================================
    # list targets
    # ========================================================

    if args.list_targets:

        print_targets()

        return

    # ========================================================
    # Validate arguments
    # ========================================================

    ratio_sum = (
        args.train_ratio
        + args.val_ratio
        + args.test_ratio
    )

    if ratio_sum > 1.0 + 1e-12:

        raise ValueError(
            "train_ratio + val_ratio + "
            "test_ratio must not be greater than 1."
        )

    output_root = Path(
        args.output_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load dataset
    # ========================================================

    dataset = load_jarvis_2021()

    # ========================================================
    # Save the raw historical dataset
    # ========================================================

    if not args.no_save_raw:

        raw_dir = (
            output_root
            / "raw"
        )

        raw_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        raw_path = (
            raw_dir
            / "jarvis_dft_3d_2021.json.gz"
        )

        print()
        print(
            "Saving raw JARVIS-DFT 2021:"
        )
        print(
            raw_path.resolve()
        )

        save_json_gz(
            raw_path,
            dataset,
        )

    # ========================================================
    # targets
    # ========================================================

    if args.target.lower() == "all":

        targets = list(
            ALIGNN_REGRESSION_TARGETS.keys()
        )

    else:

        target = args.target

        if target not in ALIGNN_REGRESSION_TARGETS:

            # Allow other continuous variables present in the database,
            # but first verify that the field exists.
            found = any(
                target in item
                for item in dataset[:100]
            )

            if not found:

                print()
                print(
                    f"Unknown target: {target}"
                )

                print_targets()

                sys.exit(1)

            print()
            print(
                "[WARNING] "
                f"{target} is not included in the predefined "
                "29 ALIGNN 2021 regression targets, "
                "but the field exists in the database; continuing."
            )

        targets = [
            target
        ]

    # ========================================================
    # Execute splits
    # ========================================================

    summaries = []

    for target in targets:

        metadata = process_target(
            dataset=dataset,
            target=target,
            output_root=output_root,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            save_records=args.save_records,
        )

        if metadata is not None:

            summaries.append(
                metadata
            )

    # ========================================================
    # summary
    # ========================================================

    summary_path = (
        output_root
        / "split_summary.csv"
    )

    save_summary(
        summary_path,
        summaries,
    )

    print()
    print("=" * 90)
    print("Finished")
    print("=" * 90)

    print()
    print(
        f"Output directory:\n"
        f"{output_root.resolve()}"
    )

    print()
    print(
        f"Summary:\n"
        f"{summary_path.resolve()}"
    )

    print()
    print(
        "The most important reproduction file under each target is:"
    )

    print(
        "    ids_train_val_test.json"
    )

    print()


if __name__ == "__main__":
    main()