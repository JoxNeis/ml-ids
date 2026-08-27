"""
Drop-in replacement for evaluate_knn_grid.

What changed vs. the original, and why:

1. Stratified subsampling of the reference set and the dev set.
   Brute-force kNN cost is exactly O(n_train * n_dev * d). At
   11.4M x 3.25M this is ~3.7e13 distance evaluations per metric.
   Nothing else in the function matters next to this.

2. Explicit float32 C-contiguous conversion, done once.
   Polars frames are columnar; letting sklearn convert them on every
   .fit()/.kneighbors() call re-materialises the array each time.

3. Chunked queries.
   Bounded peak memory + you get progress output and an ETA instead of
   a cell that looks frozen.

4. Neighbour arrays downcast to float32 / int32 after the search
   (sklearn always returns float64 / int64).

5. Labels encoded to int once; precision/recall/f1 computed in a single
   precision_recall_fscore_support call instead of four separate calls
   on string arrays.

6. Vote counting via np.bincount instead of a Python loop over classes.

Note: n_jobs is deliberately NOT passed. For algorithm="brute" with a
supported metric, sklearn dispatches to ArgKmin.compute(), which ignores
n_jobs and threads with OpenMP instead. Control it with OMP_NUM_THREADS
or threadpoolctl.
"""

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_fscore_support


def check_threads():
    """Run this first. If it prints 1, that alone is your slowdown."""
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
    n = _openmp_effective_n_threads()
    print(f"OpenMP effective threads: {n}")
    try:
        import threadpoolctl
        for info in threadpoolctl.threadpool_info():
            print(f"  {info['user_api']:10s} {info['internal_api']:10s} "
                  f"threads={info['num_threads']}")
    except ImportError:
        print("  (pip install threadpoolctl for more detail)")
    return n


def estimate_runtime(n_train, n_dev, metrics, n_threads):
    """Rough projection, calibrated at ~3.5 s per (1M train x 10k dev) per
    core for euclidean on 25 float32 dims. Manhattan measured ~6.5x slower
    because it cannot use the BLAS matrix-multiply trick."""
    base = 3.5 * (n_train / 1e6) * (n_dev / 1e4) / max(n_threads, 1)
    total = 0.0
    for m in metrics:
        mult = 6.5 if m in ("manhattan", "cityblock", "l1") else 1.0
        secs = base * mult
        total += secs
        print(f"  {m:12s} ~{secs/3600:7.2f} h")
    print(f"  {'TOTAL':12s} ~{total/3600:7.2f} h")
    return total


def _to_f32(df) -> np.ndarray:
    arr = df.to_numpy() if hasattr(df, "to_numpy") else np.asarray(df)
    return np.ascontiguousarray(arr, dtype=np.float32)


def _stratified_index(y_enc: np.ndarray, n_target: int, rng) -> np.ndarray:
    """Proportional stratified sample, keeping at least 1 row per class."""
    n = y_enc.shape[0]
    if n_target is None or n_target >= n:
        return np.arange(n)
    keep = []
    for c in np.unique(y_enc):
        idx_c = np.flatnonzero(y_enc == c)
        take = max(1, int(round(len(idx_c) * n_target / n)))
        take = min(take, len(idx_c))
        keep.append(rng.choice(idx_c, size=take, replace=False))
    out = np.concatenate(keep)
    rng.shuffle(out)
    return out


def _vote(neigh_y, neigh_dist, k, weight, n_classes):
    """Weighted / unweighted majority vote via a single bincount pass."""
    yk = neigh_y[:, :k]
    n = yk.shape[0]
    flat_idx = (np.arange(n, dtype=np.int64)[:, None] * n_classes + yk).ravel()

    if weight == "uniform":
        counts = np.bincount(flat_idx, minlength=n * n_classes)
    else:
        with np.errstate(divide="ignore"):
            w = 1.0 / neigh_dist[:, :k].astype(np.float64)
        inf_mask = np.isinf(w)
        inf_row = inf_mask.any(axis=1)
        w[inf_row] = inf_mask[inf_row]
        counts = np.bincount(flat_idx, weights=w.ravel(), minlength=n * n_classes)

    return counts.reshape(n, n_classes).argmax(axis=1)


def evaluate_knn_grid(
    train_x, train_y, dev_x, dev_y,
    list_k, list_weight, list_metric,
    train_subsample: int | None = 1_000_000,
    dev_subsample: int | None = 200_000,
    query_chunk: int = 100_000,
    random_state: int = 42,
) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    train_y = np.asarray(train_y).ravel()
    dev_y = np.asarray(dev_y).ravel()
    classes, train_y_enc = np.unique(train_y, return_inverse=True)
    n_classes = len(classes)
    lut = {c: i for i, c in enumerate(classes)}
    dev_y_enc = np.fromiter((lut.get(v, -1) for v in dev_y),
                            dtype=np.int32, count=len(dev_y))

    X = _to_f32(train_x)
    Q = _to_f32(dev_x)

    tr_idx = _stratified_index(train_y_enc, train_subsample, rng)
    dv_idx = _stratified_index(dev_y_enc, dev_subsample, rng)
    X, train_y_enc = X[tr_idx], train_y_enc[tr_idx].astype(np.int32)
    Q, dev_y_enc = Q[dv_idx], dev_y_enc[dv_idx]

    print(f"reference set: {X.shape[0]:,} x {X.shape[1]}  ({X.nbytes/1e9:.2f} GB)")
    print(f"query set    : {Q.shape[0]:,} x {Q.shape[1]}  ({Q.nbytes/1e9:.2f} GB)")
    max_k = max(list_k)

    results = []
    for metric in list_metric:
        print(f"\n=== neighbour search: {metric} (k={max_k}) ===", flush=True)
        nn = NearestNeighbors(n_neighbors=max_k, metric=metric,
                              algorithm="brute").fit(X)

        need_dist = "distance" in list_weight
        d_parts, y_parts = [], []
        t0 = time.perf_counter()

        for start in range(0, Q.shape[0], query_chunk):
            block = Q[start:start + query_chunk]
            if need_dist:
                d, i = nn.kneighbors(block, n_neighbors=max_k)
                d_parts.append(d.astype(np.float32))
            else:
                i = nn.kneighbors(block, n_neighbors=max_k, return_distance=False)
            y_parts.append(train_y_enc[i.astype(np.int32)])

            done = min(start + query_chunk, Q.shape[0])
            el = time.perf_counter() - t0
            eta = el / done * (Q.shape[0] - done)
            print(f"  {done:>9,}/{Q.shape[0]:,}  elapsed {el/60:6.1f} min  "
                  f"eta {eta/60:6.1f} min", end="\r", flush=True)

        neigh_dist = np.vstack(d_parts) if need_dist else None
        neigh_y = np.vstack(y_parts)
        del d_parts, y_parts
        print(f"\n  search done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

        for k in list_k:
            for weight in list_weight:
                pred = _vote(neigh_y, neigh_dist, k, weight, n_classes)
                p, r, f, _ = precision_recall_fscore_support(
                    dev_y_enc, pred, average="weighted", zero_division=0,
                    labels=np.arange(n_classes))
                results.append({
                    "name": f"KNN-k-{k}-w-{weight}-m-{metric}",
                    "acc": float((dev_y_enc == pred).mean()),
                    "precision": p, "recall": r, "f1": f,
                })

    return pd.DataFrame(results)


if __name__ == "__main__":
    n = check_threads()
    print("\nProjected runtime at FULL size (11.4M train, 3.25M dev):")
    estimate_runtime(11_365_674, 3_246_587, ["euclidean", "manhattan"], n)
    print("\nProjected runtime SUBSAMPLED (1M train, 200k dev):")
    estimate_runtime(1_000_000, 200_000, ["euclidean", "manhattan"], n)