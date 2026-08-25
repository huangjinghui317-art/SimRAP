#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
add_cif_to_alignn_splits.py

Purpose:
    Using the JIDs in train.csv / val.csv / test.csv / unused.csv,
    find the corresponding structures in the JARVIS-DFT 2021 dataset,
    convert them to CIF strings,
    and add them as a new cif column.

Example input CSV:

split,filtered_index,original_index,jid,target_name,target_value
train,0,123,JVASP-12870,bulk_modulus_kv,95.2

After processing:

split,filtered_index,original_index,jid,target_name,target_value,cif
train,0,123,JVASP-12870,bulk_modulus_kv,95.2,"data_..."

Features:
1. Uses JARVIS-DFT 2021: dft_3d_2021
2. Does not access individual JVASP web pages
3. Matches structures locally by jid
4. Writes CIF strings directly into CSV cells
5. Overwrites the original CSV by default
6. Optionally saves as *_with_cif.csv
7. Uses a tqdm progress bar
8. Matching failures do not interrupt execution; errors are recorded

Usage (about 30 seconds per target):

python data/add_cif_to_alignn_splits.py \
  --split-dir data/jarvis2021/targets/bulk_modulus_kv

other examples: shear_modulus_gv, mbj_bandgap, optb88vdw_bandgap, magmom_oszicar

"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms


# ============================================================
# JARVIS Atoms -> CIF string
# ============================================================

def atoms_dict_to_cif(atoms_dict):
    """
    Convert an atoms dictionary from the JARVIS database into a CIF string.

    Pipeline:
        JARVIS Atoms
             ↓
        pymatgen Structure
             ↓
        CIF string

    symprec=None:
        does not re-detect or force a space-group modification,
        so the original unit cell and atomic coordinates from the database are preserved as much as possible.
    """

    atoms = Atoms.from_dict(atoms_dict)

    pmg_structure = atoms.pymatgen_converter()

    # pymatgen Structure can directly output a CIF string
    cif_string = pmg_structure.to(
        fmt="cif"
    )

    return cif_string


# ============================================================
# Build JID -> CIF mapping
# ============================================================

def build_jid_cif_map(dataset, required_jids):
    """
    Convert only the JIDs actually required by the CSV files,
    avoiding CIF generation for all 55,723 structures.
    """

    required_jids = set(
        str(x).strip()
        for x in required_jids
    )

    print()
    print("=" * 80)
    print("Building JID -> CIF map")
    print("=" * 80)

    print(
        f"JIDs to match: {len(required_jids):,}"
    )

    jid_to_cif = {}

    failed = {}

    matched = 0

    for item in tqdm(
        dataset,
        desc="Scanning JARVIS-DFT 2021",
        unit="material",
    ):

        jid = str(
            item.get("jid", "")
        ).strip()

        # This structure does not belong to the required splits
        if jid not in required_jids:
            continue

        try:

            atoms_dict = item.get(
                "atoms"
            )

            if atoms_dict is None:
                raise ValueError(
                    "atoms field is missing"
                )

            cif = atoms_dict_to_cif(
                atoms_dict
            )

            jid_to_cif[jid] = cif

            matched += 1

        except Exception as exc:

            failed[jid] = (
                f"{type(exc).__name__}: "
                f"{exc}"
            )

        # All required JIDs have already been found,
        # so there is no need to continue scanning the dataset
        if (
            matched + len(failed)
            >= len(required_jids)
        ):
            break

    return jid_to_cif, failed


# ============================================================
# Collect all JIDs required by the CSV files
# ============================================================

def collect_required_jids(
    split_dir,
    split_names,
):

    required = set()

    existing_files = []

    for split in split_names:

        csv_path = (
            split_dir
            / f"{split}.csv"
        )

        if not csv_path.exists():

            print(
                f"[SKIP] Not found: {csv_path}"
            )

            continue

        df = pd.read_csv(
            csv_path
        )

        if "jid" not in df.columns:

            raise ValueError(
                f"{csv_path} does not contain a jid column."
            )

        jids = (
            df["jid"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        required.update(
            jids
        )

        existing_files.append(
            (
                split,
                csv_path,
            )
        )

    return (
        required,
        existing_files,
    )


# ============================================================
# Add CIF strings to one CSV
# ============================================================

def process_csv(
    csv_path,
    jid_to_cif,
    failed,
    overwrite=True,
):

    print()
    print(
        f"Processing: {csv_path}"
    )

    df = pd.read_csv(
        csv_path
    )

    if "jid" not in df.columns:

        raise ValueError(
            f"{csv_path} does not contain a jid column."
        )

    # --------------------------------------------------------
    # Write CIF strings according to jid
    # --------------------------------------------------------

    df["cif"] = (
        df["jid"]
        .astype(str)
        .str.strip()
        .map(jid_to_cif)
    )

    # --------------------------------------------------------
    # Add structure-retrieval status for troubleshooting
    # --------------------------------------------------------

    def get_status(jid):

        jid = str(jid).strip()

        if jid in jid_to_cif:
            return "ok"

        if jid in failed:
            return "cif_failed"

        return "jid_not_found"

    def get_error(jid):

        jid = str(jid).strip()

        return failed.get(
            jid,
            ""
        )

    df["cif_status"] = (
        df["jid"]
        .apply(get_status)
    )

    df["cif_error"] = (
        df["jid"]
        .apply(get_error)
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    if overwrite:

        output_path = csv_path

    else:

        output_path = (
            csv_path.parent
            / (
                csv_path.stem
                + "_with_cif.csv"
            )
        )

    df.to_csv(
        output_path,
        index=False,
    )

    ok_count = int(
        (
            df["cif_status"]
            == "ok"
        ).sum()
    )

    failed_count = (
        len(df)
        - ok_count
    )

    print(
        f"  rows   : {len(df):,}"
    )

    print(
        f"  CIF OK : {ok_count:,}"
    )

    print(
        f"  failed : {failed_count:,}"
    )

    print(
        f"  saved  : {output_path}"
    )

    return {
        "file": str(output_path),
        "rows": len(df),
        "ok": ok_count,
        "failed": failed_count,
    }


# ============================================================
# main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Add CIF strings to ALIGNN-style "
            "JARVIS-DFT 2021 split CSV files."
        )
    )

    parser.add_argument(
        "--split-dir",
        type=str,
        required=True,
        help=(
            "Directory containing train.csv / val.csv / "
            "test.csv / unused.csv."
        ),
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help=(
            "Do not overwrite the original CSV; "
            "instead, write *_with_cif.csv."
        ),
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=[
            "train",
            "val",
            "test",
            "unused",
        ],
        help=(
            "Splits to process."
        ),
    )

    args = parser.parse_args()

    split_dir = Path(
        args.split_dir
    )

    if not split_dir.exists():

        raise FileNotFoundError(
            f"Directory does not exist: {split_dir}"
        )

    print()
    print("=" * 80)
    print("Add CIF to ALIGNN split CSV")
    print("=" * 80)

    print(
        f"Split directory: "
        f"{split_dir.resolve()}"
    )

    # ========================================================
    # 1. Read CSV files first to determine which JIDs are required
    # ========================================================

    (
        required_jids,
        existing_files,
    ) = collect_required_jids(
        split_dir=split_dir,
        split_names=args.splits,
    )

    if not required_jids:

        raise RuntimeError(
            "No valid JIDs were found."
        )

    print()
    print(
        f"Unique JIDs required: "
        f"{len(required_jids):,}"
    )

    # ========================================================
    # 2. Load the exact same JARVIS-DFT 2021 version
    # ========================================================

    print()
    print("=" * 80)
    print("Loading JARVIS-DFT 2021")
    print("=" * 80)

    dataset = data(
        "dft_3d_2021"
    )

    print(
        f"Dataset size: "
        f"{len(dataset):,}"
    )

    # ========================================================
    # 3. Generate CIF strings from JIDs
    # ========================================================

    (
        jid_to_cif,
        failed,
    ) = build_jid_cif_map(
        dataset=dataset,
        required_jids=required_jids,
    )

    found_jids = (
        set(jid_to_cif.keys())
        | set(failed.keys())
    )

    missing_jids = (
        required_jids
        - found_jids
    )

    print()
    print("=" * 80)
    print("JID matching result")
    print("=" * 80)

    print(
        f"Required     : "
        f"{len(required_jids):,}"
    )

    print(
        f"CIF generated: "
        f"{len(jid_to_cif):,}"
    )

    print(
        f"CIF failed   : "
        f"{len(failed):,}"
    )

    print(
        f"JID not found: "
        f"{len(missing_jids):,}"
    )

    # ========================================================
    # 4. Add CIF strings to train / val / test splits
    # ========================================================

    summaries = []

    for split, csv_path in existing_files:

        result = process_csv(
            csv_path=csv_path,
            jid_to_cif=jid_to_cif,
            failed=failed,
            overwrite=(
                not args.no_overwrite
            ),
        )

        summaries.append(
            result
        )

    # ========================================================
    # 5. Save JIDs that were not found
    # ========================================================

    if missing_jids:

        missing_path = (
            split_dir
            / "missing_jids.txt"
        )

        with open(
            missing_path,
            "w",
            encoding="utf-8",
        ) as f:

            for jid in sorted(
                missing_jids
            ):

                f.write(
                    jid + "\n"
                )

        print()
        print(
            f"Missing JIDs saved to: "
            f"{missing_path}"
        )

    # ========================================================
    # 6. Save CIF conversion failures
    # ========================================================

    if failed:

        failed_path = (
            split_dir
            / "cif_failed.csv"
        )

        failed_df = pd.DataFrame(
            [
                {
                    "jid": jid,
                    "error": error,
                }
                for jid, error
                in failed.items()
            ]
        )

        failed_df.to_csv(
            failed_path,
            index=False,
        )

        print(
            f"CIF errors saved to: "
            f"{failed_path}"
        )

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)


if __name__ == "__main__":
    main()