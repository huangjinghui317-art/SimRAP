#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""General utility, logging, statistics, serialization, and resource-monitor blocks."""

import csv
import hashlib
import json
import logging
import math
import os
import pickle
import statistics
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import psutil


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def safe_scalar(value: Any) -> Any:
    """Convert pandas/numpy scalar to a pickle/csv friendly Python scalar."""
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def file_fingerprint(path: Path) -> Dict[str, Any]:
    st = path.stat()
    return {
        "path": resolve_path(path),
        "size_bytes": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def short_hash(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def setup_logger(name: str, log_path: Path, append: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    total: float = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        self.count += 1
        self.total += x
        self.minimum = min(self.minimum, x)
        self.maximum = max(self.maximum, x)
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def to_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "total": 0.0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        return {
            "count": self.count,
            "total": self.total,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
        }


class ResourceMonitor:
    """Background sampler for process/system CPU and RSS memory."""

    def __init__(self, interval: float = 0.5, session_id: Optional[str] = None):
        self.interval = max(0.1, float(interval))
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.process = psutil.Process(os.getpid())
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, Any]] = []
        self.logical_cpu_count = psutil.cpu_count(logical=True) or 1

    def start(self) -> None:
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                raw_proc = float(self.process.cpu_percent(interval=None))
                sys_cpu = float(psutil.cpu_percent(interval=None))
                rss_mb = float(self.process.memory_info().rss / (1024 ** 2))
                self.samples.append(
                    {
                        "session_id": self.session_id,
                        "timestamp": now_iso(),
                        "process_cpu_percent_raw": raw_proc,
                        "process_cpu_percent_machine_normalized": raw_proc / self.logical_cpu_count,
                        "system_cpu_percent": sys_cpu,
                        "rss_mb": rss_mb,
                    }
                )
            except Exception:
                pass

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval * 2))

    def summary(self) -> Dict[str, Any]:
        if not self.samples:
            return {
                "sample_count": 0,
                "logical_cpu_count": self.logical_cpu_count,
            }

        def vals(key: str) -> List[float]:
            return [float(x[key]) for x in self.samples]

        proc = vals("process_cpu_percent_raw")
        proc_norm = vals("process_cpu_percent_machine_normalized")
        sysc = vals("system_cpu_percent")
        rss = vals("rss_mb")
        return {
            "sample_count": len(self.samples),
            "logical_cpu_count": self.logical_cpu_count,
            "process_cpu_percent_raw_mean": statistics.fmean(proc),
            "process_cpu_percent_raw_max": max(proc),
            "process_cpu_percent_machine_normalized_mean": statistics.fmean(proc_norm),
            "process_cpu_percent_machine_normalized_max": max(proc_norm),
            "system_cpu_percent_mean": statistics.fmean(sysc),
            "system_cpu_percent_max": max(sysc),
            "rss_mb_mean": statistics.fmean(rss),
            "rss_mb_max": max(rss),
        }

    def save_csv(self, path: Path, append: bool = False) -> None:
        if not self.samples:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(self.samples[0].keys())
        exists = path.exists() and path.stat().st_size > 0
        mode = "a" if append else "w"
        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not append or not exists:
                writer.writeheader()
            writer.writerows(self.samples)
