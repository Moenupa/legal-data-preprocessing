"""
dataframe -> extract features -> shuffle -> split -> nshot -> save to jsonl/json
"""

__doc__ = r"""
python -m src.preprocessing             # for zero-shot, alpaca-format data
python -m src.preprocessing --shot 5    # for 5-shot data
python -m src.preprocessing --rag       # for rag data
python -m src.preprocessing --digitonly --rag # for rag and digitonly {"output": "5"} (5 for 5 months)
"""


import argparse
import random
import re

import pandas as pd
import pycnnum

from .dataio import batch_save, read_file
from .datamapper import alpaca_mapper_legal
from .fewshot import gen_fewshot


def chinese_to_int(s: str) -> int:
    assert isinstance(s, str)

    if s.isdigit():  # all Arabic numerals
        return int(s)

    return pycnnum.cn2num(s)


def extract_months(text: str) -> int | None:
    if text.isdigit():
        return int(text)

    year_match = re.findall(
        r"有期徒刑([\d]{1,2}|[一二三四五六七八九十]{1,4})年零?([\d]{1,3}|[一二三四五六七八九十]{1,4})个?月",
        text,
    )
    if year_match:
        year, month = year_match[-1]
        return chinese_to_int(year) * 12 + chinese_to_int(month)

    year_match = re.findall(
        r"([\d]{1,2}|[一二三四五六七八九十]{1,4})年零?([\d]{1,3}|[一二三四五六七八九十]{1,4})个?月有期徒刑",
        text,
    )
    if year_match:
        return chinese_to_int(year_match[-1][0]) * 12 + chinese_to_int(
            year_match[-1][1]
        )

    month_match = re.findall(
        r"有期徒刑([\d]{1,2}|[一二三四五六七八九十]{1,4})个?月", text
    )
    if month_match:
        return chinese_to_int(month_match[-1])
    month_match = re.findall(
        r"([\d]{1,2}|[一二三四五六七八九十]{1,4})个?月有期徒刑", text
    )
    if month_match:
        return chinese_to_int(month_match[-1])

    return 0
    year_match = re.findall(
        r"([\d]{1,2}|[一二三四五六七八九十]{1,4})年零?([\d]{1,3}|[一二三四五六七八九十]{1,4})个?月",
        text,
    )
    if year_match:
        year = chinese_to_int(year_match[-1][0])
        month = chinese_to_int(year_match[-1][1])

        return 12 * year + month

    month_match = re.findall(r"([\d]{1,2}|[一二三四五六七八九十]{1,4})个?月", text)
    if month_match:
        month = chinese_to_int(month_match[-1])
        return month
    return 0


def preprocess_xlsx() -> pd.DataFrame:
    feature_file = read_file("res/Shandong_feature_414.xlsx")

    # 删掉单被告为0的行 删掉数罪并罚为1的行
    if "数罪并罚" not in feature_file:
        feature_file["数罪并罚"] = 0
    if "单被告" not in feature_file:
        feature_file["单被告"] = 1
    feature_file = feature_file[feature_file["数罪并罚"] != 1]
    feature_file = feature_file[feature_file["单被告"] != 0]
    data_file = read_file("res/Shandong.xlsx")

    # 剩下的行为最终用于模型训练的数据:
    # X为右侧“山东省”文件的本院查明和本院认为列包括的文本
    # Y为左侧“山东省 feature”文件中的有期徒刑列
    # 上面X,Y可以通过案号匹配起来，做成一个文件
    data_file = data_file[["案号", "本院查明", "本院认为", "裁判结果"]]
    feature_file = feature_file[["案号", "有期徒刑"]]
    final_file = feature_file.merge(data_file, on="案号", how="inner")

    # uncomment to mix Beijing data
    # beijing_file = read_file("res/Beijing.xlsx")
    # beijing_file["有期徒刑"] = beijing_file["裁判结果"].apply(extract_months)
    # beijing_file = beijing_file[
    #     ["案号", "有期徒刑", "本院查明", "本院认为", "裁判结果"]
    # ]
    # final_file = pd.concat([final_file, beijing_file], ignore_index=True)
    final_file["裁判结果"] = final_file["裁判结果"].apply(lambda x: x.replace(" ", ""))

    final_file = final_file[final_file["有期徒刑"] <= 36]
    final_file = final_file[final_file["有期徒刑"] >= 6]

    return final_file


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Shandong sentencing data.")
    parser.add_argument(
        "--digitonly",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use digitonly format (no text in output).",
    )
    parser.add_argument(
        "--rag",
        type=bool,
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to use RAG data.",
    )
    parser.add_argument(
        "--valsize",
        type=float,
        default=0.1,
        help="validation partition ratio.",
    )
    parser.add_argument(
        "--testsize",
        type=float,
        default=0.1,
        help="test partition ratio.",
    )
    parser.add_argument(
        "--shot",
        type=int,
        default=0,
        help="Generate N-shot data.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open("res/rag.txt") as f:
        rag = f.read().strip() + "\n\n"

    input_data = preprocess_xlsx().to_dict(orient="records")
    data_samples = [
        alpaca_mapper_legal(
            sample,
            rag=rag if args.rag else "",
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
        root_dir="shandong/alpaca"
        + ("_digitonly" if args.digitonly else "")
        + ("_rag" if args.rag else "")
        + (f"_{args.shot}shot" if args.shot else ""),
    )
