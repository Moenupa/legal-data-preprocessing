def get_instruction(rag_path: str = None) -> str:
    if rag_path is None:
        return "根据以下案件信息，判断被告人应当判处的有期徒刑月数。"

    with open(rag_path) as f:
        rag = f.read().strip() + "\n\n"

    return f"{rag}根据以上参考资料及以下案件信息，判断被告人应当判处的有期徒刑月数。"


def alpaca_mapper_legal(inp: dict, rag: str = None, digitonly: bool = False) -> dict:
    assert all(k in inp for k in ["本院查明", "本院认为", "有期徒刑"])

    out = {
        "instruction": get_instruction(rag),
        "input": f"{inp.pop('本院查明')}\n\n{inp.pop('本院认为')}\n\n判处被告人".replace(
            " ", ""
        ),
    }
    sentencing_months = inp.pop("有期徒刑")
    if digitonly:
        return (
            inp
            | out
            | {
                "output": f"{sentencing_months:d}",
            }
        )
    else:
        return (
            inp
            | out
            | {
                "output": f"{sentencing_months:d}个月有期徒刑。",
            }
        )
