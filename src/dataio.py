import os
import pickle

import pandas as pd


def read_file(file_path: str, **kwargs) -> pd.DataFrame:
    """a cached reading function for `pd.read_excel`"""
    if os.path.exists(f"{file_path}.pkl"):
        with open(f"{file_path}.pkl", "rb") as f:
            return pickle.load(f)

    data = pd.read_excel(file_path, sheet_name=0, **kwargs)
    with open(f"{file_path}.pkl", "wb") as f:
        pickle.dump(data, f)
    return data


def save_as(df: pd.DataFrame, file_path: str, **kwargs):
    df.to_json(
        file_path,
        orient="records",
        force_ascii=False,
        lines=file_path.endswith(".jsonl"),
        **kwargs,
    )


def batch_save(dataset: dict[str, list[dict]], root_dir: str) -> None:
    assert all(k in dataset for k in ["train", "validation", "test"])
    os.makedirs(root_dir, exist_ok=True)

    for partition, data in dataset.items():
        save_to = f"{root_dir}/{partition}.jsonl"
        save_as(pd.DataFrame(data), save_to)
        print(f"{partition}: {len(data)} samples -> {save_to}")
