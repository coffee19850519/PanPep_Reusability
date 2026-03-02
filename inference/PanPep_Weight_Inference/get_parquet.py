import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Count rows and label=1 in a Parquet file")
    parser.add_argument("--input", required=True, help="Path to the input Parquet file")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    total_rows = len(df)
    label1_rows = (df['Label'] == 1).sum()

    print(f"File: {args.input}")
    print(f"Total rows: {total_rows}")
    print(f"label = 1 rows: {label1_rows}")

if __name__ == "__main__":
    main()
