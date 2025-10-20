import random

import numpy as np


def dict2str(sample: dict) -> str:
    KEYSETS = [["instruction", "input", "output"], ["problem", "answer"]]
    for keys in KEYSETS:
        if all(k in sample for k in keys):
            return "\n".join(f"{sample[k]}" for k in keys)

    assert False
    return " ".join(f"{v}" for v in sample.values())


def gen_fewshot(
    train: list[dict],
    validation: list[dict],
    test: list[dict],
    n_shot: int = 0,
) -> dict[str, list[dict]]:
    # generate few-shot -> train, val, test
    if n_shot == 0:
        return {
            "train": train,
            "validation": validation,
            "test": test,
        }

    # ensure n_shot is not larger than available training examples
    n_shot = min(n_shot, len(train))

    # convert list[dict] -> list[str]
    train_texts = [dict2str(x) for x in train]
    val_texts = [dict2str(x) for x in validation]
    test_texts = [dict2str(x) for x in test]

    use_st = False
    train_emb = val_emb = test_emb = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        train_emb = model.encode(train_texts, convert_to_numpy=True)
        val_emb = model.encode(val_texts, convert_to_numpy=True) if val_texts else None
        test_emb = (
            model.encode(test_texts, convert_to_numpy=True) if test_texts else None
        )
        use_st = True
    except Exception as e:
        print(e)
        random.seed(0)

    def get_shots_by_embedding(query_emb, train_emb_matrix, train_list):
        sims = (train_emb_matrix @ query_emb) / (
            np.linalg.norm(train_emb_matrix, axis=1)
            * (np.linalg.norm(query_emb) + 1e-12)
        )
        idx = sims.argsort()[-n_shot:][::-1]
        return [train_list[i] for i in idx]

    def prepend_shots_to_item(item: dict, shots: list[dict]) -> dict:
        new_item = dict(item)  # shallow copy
        # alpaca/math12k style
        if "instruction" in item and "input" in item and "output" in item:
            shot_texts = []
            for s in shots:
                shot_texts.append(f"{s.get('input', '')}{s.get('output', '')}")
            new_item["input"] = "\n\n".join(shot_texts) + "\n\n" + item.get("input", "")
            return new_item
        # easyr1 style
        if "problem" in item and "answer" in item:
            shot_texts = []
            for s in shots:
                shot_texts.append(
                    f"{s.get('problem', '')}\n答案：{s.get('answer', '')}"
                )
            new_item["problem"] = (
                "\n\n".join(shot_texts) + "\n\n" + item.get("problem", "")
            )
            return new_item
        # generic fallback: put into 'input' key
        shot_texts = []
        for s in shots:
            shot_texts.append(" ".join(str(v) for v in s.values()))
        base = item.get("input", " ".join(str(v) for v in item.values()))
        new_item["input"] = "\n\n".join(shot_texts) + "\n\n" + base
        return new_item

    # build new validation and test sets with few-shot prefixes
    new_validation = []
    for i, itm in enumerate(validation):
        if use_st and val_emb is not None:
            shots = get_shots_by_embedding(val_emb[i], train_emb, train)
        else:
            print("warn: using random shots for validation")
            shots = random.sample(train, n_shot)
        new_validation.append(prepend_shots_to_item(itm, shots))

    new_test = []
    for i, itm in enumerate(test):
        if use_st and test_emb is not None:
            shots = get_shots_by_embedding(test_emb[i], train_emb, train)
        else:
            print("warn: using random shots for test")
            shots = random.sample(train, n_shot)
        new_test.append(prepend_shots_to_item(itm, shots))

    return {"train": train, "validation": new_validation, "test": new_test}
