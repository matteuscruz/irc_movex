#!/usr/bin/env python3
"""
Split a Parquet file into N parts (default 2) and create a ZIP for each part.

Usage:
  python scripts/split_parquet_and_zip.py --input data/MWC_v3_processed.parquet --parts 2

This will create files next to the input file:
  data/MWC_v3_processed_part1.parquet
  data/MWC_v3_processed_part1.zip
  data/MWC_v3_processed_part2.parquet
  data/MWC_v3_processed_part2.zip

The script requires pandas (and pyarrow or fastparquet) to read/write parquet.
"""
import argparse
from pathlib import Path
import math
import shutil
import sys

def split_parquet(input_path: Path, parts: int = 2, remove_parts: bool = False):
    import pandas as pd

    if not input_path.exists():
        print(f"Arquivo não encontrado: {input_path}")
        sys.exit(1)

    print(f"Lendo {input_path} ...")
    df = pd.read_parquet(input_path)
    n = len(df)
    if n == 0:
        print("Arquivo vazio. Nada a fazer.")
        return []

    part_size = math.ceil(n / parts)
    out_files = []

    for i in range(parts):
        start = i * part_size
        end = min(start + part_size, n)
        df_part = df.iloc[start:end]
        out_parquet = input_path.with_name(f"{input_path.stem}_part{i+1}{input_path.suffix}")
        print(f"Escrevendo {out_parquet} ({len(df_part)} linhas) ...")
        df_part.to_parquet(out_parquet, index=False)
        out_zip = out_parquet.with_suffix('.zip')
        print(f"Compactando {out_parquet} -> {out_zip} ...")
        shutil.make_archive(str(out_parquet.with_suffix('')), 'zip', root_dir=out_parquet.parent, base_dir=out_parquet.name)
        out_files.append((out_parquet, out_zip))
        if remove_parts:
            out_parquet.unlink()

    return out_files


def main():
    p = argparse.ArgumentParser(description="Split a parquet into N parts and zip each part")
    p.add_argument('--input', '-i', required=True, help='Input parquet file path')
    p.add_argument('--parts', '-p', type=int, default=2, help='Number of parts (default 2)')
    p.add_argument('--remove-parts', action='store_true', help='Remove generated parquet part files after zipping')
    args = p.parse_args()

    input_path = Path(args.input)
    parts = max(1, args.parts)

    result = split_parquet(input_path, parts=parts, remove_parts=args.remove_parts)
    print('\nResultado:')
    for parquet, zipf in result:
        print(f"  {parquet} -> {zipf}")


if __name__ == '__main__':
    main()
