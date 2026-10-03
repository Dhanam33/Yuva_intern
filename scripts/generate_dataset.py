#!/usr/bin/env python3
"""Regenerate the deterministic synthetic input dataset."""

from yuva_ml.dataset import write_raw_dataset


if __name__ == "__main__":
    frame = write_raw_dataset()
    print(f"Wrote {len(frame):,} rows to data/customer_churn_raw.csv")
    print(f"Churn prevalence: {(frame['churn'] == 'Yes').mean():.1%}")
    print(f"Missing feature cells: {int(frame.drop(columns=['customer_id', 'churn']).isna().sum().sum()):,}")
