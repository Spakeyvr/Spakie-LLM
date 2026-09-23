"""Reject known lossy downloaded code before building new training assets."""
import json
from pathlib import Path

LOSSLESS_PYTHON_EDU_SCHEMA = 4


def mark_retained_lossy_code(progress: dict, *, source_dir: Path | None = None) -> None:
    if int(progress.get('python_edu_filter_schema_version', 0)) >= LOSSLESS_PYTHON_EDU_SCHEMA:
        return
    if (int(progress.get('docs_written', 0)) > 0 or
            (source_dir is not None and any(source_dir.glob('*.jsonl')))):
        progress['python_edu_requires_fresh_download'] = True


def validate_raw_code_provenance(raw_root: str | Path, *, selected_paths=None) -> None:
    root = Path(raw_root)
    candidates = [root/'python_edu', root/'large_corpus/python_edu']
    if root.name == 'python_edu':
        candidates.append(root)
    for source in candidates:
        if selected_paths is not None and not any(
                Path(path).resolve().is_relative_to(source.resolve()) for path in selected_paths):
            continue
        progress_path = source/'progress.json'
        if not progress_path.is_file():
            continue  # Hand-curated inputs need not have downloader metadata.
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        has_output = int(progress.get('docs_written', 0)) > 0 or any(source.glob('*.jsonl'))
        if has_output and (progress.get('python_edu_requires_fresh_download') or
                int(progress.get('python_edu_filter_schema_version', 0)) < LOSSLESS_PYTHON_EDU_SCHEMA):
            raise ValueError(
                f'Python-Edu at {source} contains output from the old lossy whitespace cleaner. '
                'Download this source into a new directory with the current downloader before '
                'training a new tokenizer or preparing a new corpus. --resume preserves the old '
                'rows and cannot repair string literals. Keep the original assets for old checkpoints.'
            )
