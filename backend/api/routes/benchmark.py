"""Benchmark endpoints: GET /benchmark-datasets, GET /benchmark-results."""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.models import BenchmarkResult

logger = logging.getLogger(__name__)
router = APIRouter()

_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def _dataset_key(path: Path) -> str:
    """Extract the dataset key from a benchmark results filename."""
    return path.stem.removeprefix("benchmark_results_")


def _dataset_label(key: str) -> str:
    """Convert a dataset key to a human-readable label."""
    return key.replace("_", " ").title()


def _sorted_result_files() -> list[Path]:
    """Return benchmark result files sorted newest-first by modification time."""
    return sorted(
        _DATA_DIR.glob("benchmark_results_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _available_datasets() -> list[dict]:
    """List all benchmark result files sorted by modification time."""
    return [{"key": _dataset_key(p), "label": _dataset_label(_dataset_key(p))} for p in _sorted_result_files()]


@router.get("/benchmark-datasets")
async def get_benchmark_datasets() -> list[dict]:
    """List available benchmark datasets by scanning data/benchmark_results_*.json files."""
    datasets = _available_datasets()
    if not datasets:
        raise HTTPException(status_code=404, detail="No benchmark results found. Run: python -m backend.tools.benchmark")
    return datasets


@router.get("/benchmark-results", response_model=list[BenchmarkResult])
async def get_benchmark_results(dataset: str | None = None) -> list[BenchmarkResult]:
    """Return saved benchmark results for the given dataset key.

    If no dataset is specified, returns the most recently generated file.
    Run ``python -m backend.tools.benchmark`` first to generate files.
    """
    if dataset:
        results_path = _DATA_DIR / f"benchmark_results_{dataset}.json"
        if not results_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")
    else:
        candidates = _sorted_result_files()
        if not candidates:
            raise HTTPException(status_code=404, detail="No benchmark results found. Run: python -m backend.tools.benchmark")
        results_path = candidates[0]
    with open(results_path) as f:
        return [BenchmarkResult(**row) for row in json.load(f)]
