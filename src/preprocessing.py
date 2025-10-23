"""
dataframe -> extract features -> shuffle -> split -> nshot -> save to jsonl/json
"""

__doc__ = r"""
python -m src.preprocessing                 # for 0-shot, alpaca-format data
python -m src.preprocessing --shot 3        # for 3-shot data
python -m src.preprocessing --rag /path/to/rag  # for rag data
python -m src.preprocessing --digitonly     # {"output": "5"} (5 months)
"""


import argparse
import os.path as osp
import random

import pandas as pd

from .cnhandler import extract_months
from .dataio import batch_save, read_file
from .datamapper import alpaca_mapper_legal
from .fewshot import gen_fewshot


def process(province: str, filters: list[str] = None) -> pd.DataFrame:
    filters = set(filters or [])

    match province.lower():
        case "shandong":
            # 剩下的行为最终用于模型训练的数据:
            # X为右侧'山东省'文件的本院查明和本院认为列包括的文本
            # Y为左侧'山东省 feature'文件中的有期徒刑列
            feature_file = read_file("res/Shandong_feature_414.xlsx")
            data_file = read_file("res/Shandong.xlsx")

            # 上面X,Y可以通过案号匹配起来，做成一个文件
            feature_file = feature_file[["案号", "有期徒刑", "数罪并罚", "单被告"]]
            data_file = data_file[["案号", "本院查明", "本院认为", "裁判结果"]]
            out = feature_file.merge(data_file, on="案号", how="inner")
        case "beijing":
            beijing_file = read_file("res/Beijing.xlsx")
            beijing_file["有期徒刑"] = beijing_file["裁判结果"].apply(extract_months)
            out = beijing_file
        case _:
            raise NotImplementedError(f"{province}: unsupported province")

    assert isinstance(out, pd.DataFrame)
    OUTPUT_COLUMNS = ["案号", "有期徒刑", "本院查明", "本院认为", "裁判结果"]
    assert set(OUTPUT_COLUMNS) <= set(out.columns)

    if "数罪并罚" not in out:
        out["数罪并罚"] = 0
    if "单被告" not in out:
        out["单被告"] = 1

    if "单被告" in filters:
        out = out[out["单被告"] == 1]
    if "数罪并罚" in filters:
        out = out[out["数罪并罚"] == 0]
    if "有期徒刑范围" in filters:
        # 有期徒刑在6-36个月之间
        out = out[(out["有期徒刑"] >= 6) & (out["有期徒刑"] <= 36)]
    else:
        # basic filter
        out = out[out["有期徒刑"] >= 6]

    # cleaning
    out["裁判结果"] = out["裁判结果"].apply(lambda x: x.replace(" ", ""))
    return out[OUTPUT_COLUMNS]


def batch_process(provinces: list[str], filters: list[str]) -> pd.DataFrame:
    out = process(provinces[0], filters=filters)
    for province in provinces[1:]:
        out = pd.concat([out, process(province, filters=filters)], ignore_index=True)
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Shandong sentencing data.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="out",
        help="Output directory for processed files.",
    )
    parser.add_argument(
        "--province",
        type=str,
        nargs="+",
        default=["shandong"],
        help="Province(s) to process.",
    )
    parser.add_argument(
        "--filters",
        type=str,
        nargs="+",
        default=["单被告", "数罪并罚", "有期徒刑范围"],
        help="Filters to apply.",
    )
    parser.add_argument(
        "--digitonly",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use digitonly format (no text in output).",
    )
    parser.add_argument(
        "--rag",
        type=str,
        default=None,
        help="Path to RAG data.",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=0,
        help="Generate N-shot data.",
    )
    parser.add_argument(
        "--valsize",
        type=float,
        default=0.1,
        help="Validation partition ratio.",
    )
    parser.add_argument(
        "--testsize",
        type=float,
        default=0.1,
        help="Test partition ratio.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    input_data = batch_process(args.province, filters=args.filters).to_dict(
        orient="records"
    )
    data_samples = [
        alpaca_mapper_legal(
            sample,
            rag=args.rag,
            digitonly=args.digitonly,
        )
        for sample in input_data
    ]

    # deterministic shuffle
    random.seed(0)
    random.shuffle(data_samples)

    val_split_at = int(len(data_samples) * args.valsize)
    test_split_at = int(len(data_samples) * (args.valsize + args.testsize))

    # mutually exclusive splits, 8:1:1
    dataset = gen_fewshot(
        **{
            "validation": data_samples[:val_split_at],
            "test": data_samples[val_split_at:test_split_at],
            "train": data_samples[test_split_at:],
        },
        n_shot=args.shot,
    )

    batch_save(
        dataset,
        root_dir=f"{args.outdir}/alpaca"
        + ("_digitonly" if args.digitonly else "")
        + (f"_{osp.basename(args.rag).split('.')[0]}" if args.rag is not None else "")
        + (f"_{args.shot}shot" if args.shot else ""),
    )
