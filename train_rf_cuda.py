"""Train the notebook's cuML GPU Random Forest on the SMOTE train split."""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from contextlib import contextmanager

import joblib
import numpy as np
import polars as pl

# Paths and constants, mirroring the notebook's definitions.
PATH_FOLDER_SMOTE = "data-smote"
PATH_FOLDER_MODEL = "trained-model"
LABEL_COLUMN = "Label"

RF_RANDOM_STATE = 42
RF_SPLIT_CRITERION = "gini"
RF_N_BINS = 128
RF_N_STREAMS = 4

# The winning combination from the notebook's search.
RF_MODEL_N_ESTIMATORS = 300
RF_MODEL_MAX_DEPTH = 24
RF_MODEL_MAX_FEATURES = "sqrt"


@contextmanager
def step(message: str):
    """Announce a blocking stage, then tick it off when it returns."""
    print(f"  {message} ...", end="", flush=True)
    started = time.time()
    try:
        yield
    except BaseException:
        print(" failed")
        raise
    print(f" done ({time.time() - started:.1f}s)")


def prepare_cuda_env() -> None:
    """Point CUDA_PATH at the wheel-installed toolkit before cuML is imported.

    Same `sys.prefix/targets/x86_64-linux` probe the notebook does in its setup cell;
    harmless when a system toolkit is already on the path, since it only fills in a
    value that is not set.
    """
    cuda_root = os.path.join(sys.prefix, "targets", "x86_64-linux")
    if os.path.isdir(os.path.join(cuda_root, "include")):
        os.environ.setdefault("CUDA_PATH", cuda_root)


def get_split_parquet_files(source_folder_name: str, split_name: str) -> list[str]:
    """Return sorted paths to all Parquet files under `source_folder_name/split_name`."""
    files = sorted(glob.glob(os.path.join(source_folder_name, split_name, "*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {os.path.join(source_folder_name, split_name)}"
        )
    print(f"Found {len(files)} {split_name} files")
    return files


def load_split_arrays(
    source_folder_name: str,
    split_name: str = "train",
    label_column: str = LABEL_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream the split into a preallocated float32 matrix and int32 labels.

    Collecting every file into one DataFrame and then calling `to_numpy` holds two
    full copies of the split at once, which is what the OOM killer trips on for an
    11M-row split on a 7 GB box. Filling a preallocated array file by file keeps the
    peak at roughly the size of the final matrix plus a single file.
    """
    label_column = label_column.lower()
    files = get_split_parquet_files(source_folder_name, split_name)

    schema = pl.scan_parquet(files[0]).collect_schema()
    if label_column not in schema.names():
        raise ValueError(f"Label column '{label_column}' not found in {files[0]}.")
    feature_columns = [name for name in schema.names() if name != label_column]

    # Parquet stores row counts in its footer, so this reads metadata, not columns.
    row_counts = [
        pl.scan_parquet(file).select(pl.len()).collect().item() for file in files
    ]
    n_rows = sum(row_counts)

    features = np.empty((n_rows, len(feature_columns)), dtype=np.float32)
    labels = np.empty(n_rows, dtype=np.int32)

    offset = 0
    for file, count in zip(files, row_counts):
        # Selecting by name pins column order across files and fails loudly on a
        # file whose schema drifted, rather than silently misaligning features.
        frame = pl.read_parquet(file)
        features[offset : offset + count] = frame.select(
            pl.col(feature_columns).cast(pl.Float32)
        ).to_numpy()
        labels[offset : offset + count] = frame[label_column].to_numpy()
        offset += count
        del frame

    return features, labels


def to_features(df) -> np.ndarray:
    """polars DataFrame -> C-contiguous float32 array for cuML."""
    if isinstance(df, (pl.DataFrame, pl.Series)):
        df = df.to_numpy()
    return np.ascontiguousarray(df, dtype=np.float32)


def to_labels(s) -> np.ndarray:
    """polars Series -> int32 labels (cuML classifiers reject float labels)."""
    if isinstance(s, (pl.DataFrame, pl.Series)):
        s = s.to_numpy()
    return np.asarray(s).ravel().astype(np.int32)


def get_rf_name(n_estimators, max_depth, max_features) -> str:
    return f"rf-n-{n_estimators}-d-{max_depth}-f-{max_features}.pkl"


def build_rf(
    n_estimators: int,
    max_depth: int,
    max_features,
    split_criterion: str = RF_SPLIT_CRITERION,
    n_bins: int = RF_N_BINS,
    n_streams: int = RF_N_STREAMS,
    random_state: int = RF_RANDOM_STATE,
):
    """Construct the cuML forest.

    `n_streams > 1` builds trees on several CUDA streams whose timing is not
    deterministic, so `random_state` alone does not pin the forest exactly; pass
    `--n-streams 1` for an exactly reproducible fit, or when the card runs out of
    memory building trees concurrently.
    """
    from cuml.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        max_features=max_features,
        split_criterion=split_criterion,
        n_bins=n_bins,
        n_streams=n_streams,
        random_state=random_state,
    )


def dump_trained_model(model, file_path: str) -> str:
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    joblib.dump(model, file_path)
    return file_path


def train_rf(
    train_x,
    train_y,
    n_estimators: int,
    max_depth: int,
    max_features,
    output_path: str,
    random_state: int = RF_RANDOM_STATE,
    split_criterion: str = RF_SPLIT_CRITERION,
    n_bins: int = RF_N_BINS,
    n_streams: int = RF_N_STREAMS,
):
    print(
        f"Training Random Forest: n_estimators={n_estimators}"
        f" max_depth={max_depth} max_features={max_features}"
    )

    print("Preparing training data...")
    features = to_features(train_x)
    labels = to_labels(train_y)
    n_rows, n_features = features.shape

    model = build_rf(
        n_estimators,
        max_depth,
        max_features,
        split_criterion=split_criterion,
        n_bins=n_bins,
        n_streams=n_streams,
        random_state=random_state,
    )
    with step(f"fitting {n_estimators} trees on {n_rows:,} rows x {n_features} features"):
        model.fit(features, labels)

    with step("saving model"):
        file_path = dump_trained_model(model, output_path)
    print(f"Saved model to: {file_path}")
    return model


def parse_max_features(value: str):
    """`sqrt` / `log2` stay strings; a number is a float ratio or an int column count."""
    if value in ("sqrt", "log2", "auto"):
        return value
    number = float(value)
    return int(number) if number.is_integer() and number > 1 else number


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-folder",
        default=PATH_FOLDER_SMOTE,
        help=f"folder holding the <split>/*.parquet files (default: {PATH_FOLDER_SMOTE})",
    )
    parser.add_argument("--split", default="train", help="split subfolder (default: train)")
    parser.add_argument("--label-column", default=LABEL_COLUMN)
    parser.add_argument("--n-estimators", type=int, default=RF_MODEL_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=RF_MODEL_MAX_DEPTH)
    parser.add_argument(
        "--max-features", type=parse_max_features, default=RF_MODEL_MAX_FEATURES
    )
    parser.add_argument("--split-criterion", default=RF_SPLIT_CRITERION)
    parser.add_argument("--n-bins", type=int, default=RF_N_BINS)
    parser.add_argument(
        "--n-streams",
        type=int,
        default=RF_N_STREAMS,
        help="trees built concurrently; 1 is reproducible and uses the least memory",
    )
    parser.add_argument("--random-state", type=int, default=RF_RANDOM_STATE)
    parser.add_argument(
        "--output",
        default=None,
        help="path to dump the pickled model to"
        f" (default: {PATH_FOLDER_MODEL}/RF/<generated name>)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    prepare_cuda_env()

    output_path = args.output or os.path.join(
        PATH_FOLDER_MODEL,
        "RF",
        get_rf_name(args.n_estimators, args.max_depth, args.max_features),
    )

    started = time.time()
    print(f"Loading {args.split} split from {args.data_folder}...")
    train_x, train_y = load_split_arrays(
        args.data_folder, args.split, args.label_column
    )
    print(f"Loaded {train_x.shape[0]:,} rows x {train_x.shape[1]} features")

    train_rf(
        train_x,
        train_y,
        args.n_estimators,
        args.max_depth,
        args.max_features,
        output_path,
        random_state=args.random_state,
        split_criterion=args.split_criterion,
        n_bins=args.n_bins,
        n_streams=args.n_streams,
    )
    print(f"Total: {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
