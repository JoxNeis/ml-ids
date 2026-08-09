import os
import glob
import time
import pandas as pd

INPUT_DIR = "cse-cic-ids2018"
OUTPUT_DIR = "parquet"
CHUNK_SIZE = 100_000

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert_csv(csv_file):
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    print(f"Converting {base_name}...")

    for i, chunk in enumerate(
        pd.read_csv(csv_file, chunksize=CHUNK_SIZE, low_memory=False),
        start=1
    ):
        output = os.path.join(
            OUTPUT_DIR,
            f"{base_name}_{i:05d}.parquet"
        )

        chunk = chunk[chunk["Label"] != "Label"]

        if os.path.exists(output):
            os.remove(output)

        chunk.to_parquet(
            output,
            engine="pyarrow",
            compression="snappy",
            index=False
        )

        print(f"  Saved {output}")


if __name__ == "__main__":
    start = time.perf_counter()
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))

    for csv_file in csv_files:
        convert_csv(csv_file)

    end = time.perf_counter()
    print(f"\nFinished in {end - start:.2f} seconds")