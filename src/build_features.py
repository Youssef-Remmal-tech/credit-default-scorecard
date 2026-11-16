"""Run the SQL layer in DuckDB and write the feature mart."""
import hashlib

import duckdb
import pandas as pd

from . import config


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_feature_mart() -> pd.DataFrame:
    if not config.RAW_CSV.exists():
        raise FileNotFoundError(f"Raw data not found at {config.RAW_CSV}. See data/README.md.")
    digest = file_sha256(config.RAW_CSV)
    if digest != config.RAW_SHA256:
        raise ValueError(f"Raw file checksum {digest} does not match the expected UCI file.")

    con = duckdb.connect()
    con.execute("SET VARIABLE raw_path = ?", [str(config.RAW_CSV)])
    for name in ("01_staging.sql", "02_features.sql"):
        con.execute((config.SQL_DIR / name).read_text())

    checks = con.execute((config.SQL_DIR / "03_checks.sql").read_text()).df()
    failed = checks[checks["failures"] != 0]
    if not failed.empty:
        raise ValueError(f"Data-quality checks failed:\n{failed}")

    mart = con.execute("SELECT * FROM feature_mart").df()
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    mart.to_csv(config.PROCESSED / "feature_mart.csv", index=False)
    con.close()
    return mart


if __name__ == "__main__":
    df = build_feature_mart()
    print(f"feature_mart: {len(df):,} rows, {df.shape[1]} columns")
