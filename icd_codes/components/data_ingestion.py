import io
import requests
import polars as pl

from icd_codes.utils.config import load_config


config = load_config("config/default.yaml")
DATA_URL = config["dataset_url"]


def ingest_data() -> pl.DataFrame:
    """
    Fetch CSV from DATA_URL and load into a Polars DataFrame.
    Raises explicit errors on failure.
    """
    try:
        resp = requests.get(DATA_URL, timeout=60)
        resp.raise_for_status()

        df = pl.read_csv(io.BytesIO(resp.content))

        df = df.rename({" Note": "Note"})
        if df.is_empty():
            raise ValueError("Downloaded DataFrame is empty")

        # Save full dataset
        df.write_csv(r"data/medsynth.csv")

        # Train/Test Split
        seed = config.get("seed", 42)
        df = df.sample(fraction=1.0, shuffle=True, seed=seed)
        
        split_idx = int(len(df) * 0.8)
        train_df = df[:split_idx]
        test_df = df[split_idx:]

        train_df.write_csv("data/train.csv")
        test_df.write_csv("data/test.csv")

        return df

    except requests.RequestException as e:
        raise RuntimeError(f"HTTP error while fetching dataset: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to parse dataset: {e}") from e


if __name__ == "__main__":
    try:
        df = ingest_data()
        print(df.shape)
        print(df.head())
    except Exception as e:
        print(f"Data ingestion failed: {e}")
