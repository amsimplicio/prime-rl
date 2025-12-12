import verifiers as vf
from datasets import Dataset

def _ensure_info_dataset(dataset: Dataset) -> Dataset:
    """
    Make sure the dataset has a valid 'info' column:
      - If no 'info' column exists: create one from 'answer' if available, else {}.
      - If 'info' exists but is all None: drop it and recreate from 'answer' / {}.
      - If 'info' exists and has some None: fill just those with {} or {'answer': ...}.
    """
    if "info" not in dataset.column_names:
        answers = dataset["answer"] if "answer" in dataset.column_names else [None] * len(dataset)
        infos = [{"answer": a} if a is not None else {} for a in answers]
        return dataset.add_column("info", infos)

    col = dataset["info"]
    if len(col) == 0:
        return dataset

    all_none = all(v is None for v in col)
    any_none = any(v is None for v in col)

    # Case 1: info column exists but is entirely None
    if all_none:
        dataset = dataset.remove_columns("info")
        answers = dataset["answer"] if "answer" in dataset.column_names else [None] * len(dataset)
        infos = [{"answer": a} if a is not None else {} for a in answers]
        return dataset.add_column("info", infos)

    # Case 2: mixture of None / dicts → fill only the Nones
    if any_none:
        def fill_info(batch):
            infos = batch["info"]
            answers = batch.get("answer", [None] * len(infos))
            new_infos = []
            for info, ans in zip(infos, answers):
                if info is None:
                    d = {}
                    if ans is not None:
                        d["answer"] = ans
                    new_infos.append(d)
                else:
                    new_infos.append(info)
            batch["info"] = new_infos
            return batch

        dataset = dataset.map(fill_info, batched=True)

    return dataset


def _ensure_answer_string(dataset: Dataset) -> Dataset:
    """
    Ensure the 'answer' column is always a string (even if empty),
    because GenerateOutputs requires strings.
    """
    if "answer" not in dataset.column_names:
        dataset = dataset.add_column("answer", [""] * len(dataset))
        return dataset

    def fix(batch):
        answers = batch["answer"]
        new = []
        for a in answers:
            if a is None:
                new.append("")
            else:
                new.append(str(a))
        batch["answer"] = new
        return batch

    return dataset.map(fix, batched=True)


def _prepare_env(env):
    if getattr(env, "dataset", None) is not None:
        ds = env.dataset
        ds = _ensure_info_dataset(ds)
        ds = _ensure_answer_string(ds)
        env.dataset = ds

    if getattr(env, "eval_dataset", None) is not None:
        ds = env.eval_dataset
        ds = _ensure_info_dataset(ds)
        ds = _ensure_answer_string(ds)
        env.eval_dataset = ds

    return env


def load_environment(**kwargs):
    """
    Wrapper around hendrycks-math that fixes 'info' and 'answer'
    but DOES NOT create an EnvGroup.
    """
    base_env = vf.load_environment("hendrycks-math")
    return _prepare_env(base_env)
