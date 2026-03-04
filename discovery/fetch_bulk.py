"""
Bulk parallel notebook downloader.
Reads notebook_index.jsonl and downloads all notebooks with 30 workers.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApiExtended
from loguru import logger
from tqdm import tqdm


def download_single(
    api: KaggleApiExtended, kernel_ref: str, output_dir: Path
) -> tuple[str, bool]:
    """Download one notebook. Returns (kernel_ref, success)."""
    out_path = output_dir / f"{kernel_ref.replace('/', '__')}.ipynb"
    if out_path.exists():
        return kernel_ref, True
    try:
        api.kernels_pull(kernel_ref, path=str(output_dir), metadata=False)
        return kernel_ref, True
    except Exception as e:
        logger.debug(f"Failed {kernel_ref}: {e}")
        return kernel_ref, False


def bulk_download(
    index_path: Path,
    output_dir: Path,
    n_workers: int = 30,
    limit: int | None = None,
) -> None:
    """
    Download all notebooks listed in index_path using n_workers parallel threads.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(index_path) as f:
        entries = [json.loads(line) for line in f]

    if limit:
        entries = entries[:limit]

    kernel_refs = [e["kernel_ref"] for e in entries if e.get("kernel_ref")]
    logger.info(f"Downloading {len(kernel_refs)} notebooks with {n_workers} workers")

    api = KaggleApiExtended()
    api.authenticate()

    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(download_single, api, ref, output_dir): ref
            for ref in kernel_refs
        }
        with tqdm(total=len(futures), desc="Downloading notebooks") as pbar:
            for future in as_completed(futures):
                _, ok = future.result()
                if ok:
                    success += 1
                else:
                    failed += 1
                pbar.update(1)
                pbar.set_postfix(success=success, failed=failed)

    logger.info(f"Download complete: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    import typer

    def main(
        index: str = "./data/raw/notebooks/notebook_index.jsonl",
        output_dir: str = "./data/raw/notebooks/files",
        workers: int = 30,
        limit: int = None,
    ):
        bulk_download(Path(index), Path(output_dir), workers, limit)

    typer.run(main)
