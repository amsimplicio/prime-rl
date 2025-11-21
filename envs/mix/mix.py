import verifiers as vf
from datasets import Dataset

# Columns that are "core" to verifiers and should be shared across envs
CORE_COLS = {
    "prompt",       # main input
    "answer",       # gold answer (if any)
    "info",         # dict with answer/metadata used by rubrics
    "question",     # some envs keep the question separate from prompt
    "task",         # added by EnvGroup
    "example_id",   # optional id
}


def _ensure_info_dataset(dataset: Dataset) -> Dataset:
    """
    Make sure the dataset has a valid 'info' column:
      - If no 'info' column exists: create one from 'answer' if available, else {}.
      - If 'info' exists but is all None: drop it and recreate from 'answer' / {}.
      - If 'info' exists and has some None: fill just those with {} or {'answer': ...}.
      - If 'info' exists and is non-None everywhere: leave as is (e.g. for IF envs).
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

    # Case 3: info already fine
    return dataset

def _ensure_answer_string(dataset: Dataset) -> Dataset:
    """
    Ensure the 'answer' column is always a string (even if empty),
    because GenerateOutputs requires strings.
    """
    if "answer" not in dataset.column_names:
        # create empty-string answers if missing entirely
        dataset = dataset.add_column("answer", [""] * len(dataset))
        return dataset

    def fix(batch):
        answers = batch["answer"]
        new = []
        for a in answers:
            if a is None:
                new.append("")   # <- empty string is valid
            else:
                new.append(str(a))
        batch["answer"] = new
        return batch

    return dataset.map(fix, batched=True)


def _namespace_extra_columns(dataset: Dataset, env_name: str) -> Dataset:
    """
    Rename non-core columns to env-specific names so that features
    don't collide across envs.

    Example:
      - 'puzzle' in sudoku → 'sudoku__puzzle'
      - 'puzzle' in logic_puzzle_solver → 'logic_puzzle_solver__puzzle'

    This way, HuggingFace concatenate_datasets never has to align
    columns with incompatible types under the same name.
    """
    rename_map = {}
    for col in dataset.column_names:
        if col not in CORE_COLS:
            rename_map[col] = f"{env_name}__{col}"
    if rename_map:
        dataset = dataset.rename_columns(rename_map)
    return dataset


def _prepare_env(env, env_name: str):
    if getattr(env, "dataset", None) is not None:
        ds = env.dataset
        ds = _ensure_info_dataset(ds)
        ds = _ensure_answer_string(ds)   # <-- add here
        ds = _namespace_extra_columns(ds, env_name)
        env.dataset = ds

    if getattr(env, "eval_dataset", None) is not None:
        ds = env.eval_dataset
        ds = _ensure_info_dataset(ds)
        ds = _ensure_answer_string(ds)   # <-- add here
        ds = _namespace_extra_columns(ds, env_name)
        env.eval_dataset = ds

    return env



def load_environment(**kwargs):
    """
    Big mixed environment for prime-rl.

    You can include whichever envs you want here, e.g.:

      - vf-pt-instruction-following  (ifpt)
      - vf-en-instruction-following  (ifen)
      - hendrycks-math               (hendrycksmath)
      - gsm8k                        (gsm8k)
      - logic-puzzle-solver          (logic_puzzle_solver)
      - sudoku                       (sudoku)
      - mini-sudoku                  (mini_sudoku)

    All child envs are normalized so that:
      - 'info' is always a dict.
      - non-core columns (like 'puzzle') are env-specific and never collide.
    """

    # Load whatever envs you want in the big mix:
    ifpt                = vf.load_environment("vf-pt-instruction-following")
    ifen                = vf.load_environment("vf-en-instruction-following")
    hendrycksmath       = vf.load_environment("hendrycks-math")
    gsm8k               = vf.load_environment("gsm8k")
    logic_puzzle_solver = vf.load_environment("logic-puzzle-solver")
    sudoku              = vf.load_environment("sudoku")
    mini_sudoku         = vf.load_environment("mini-sudoku")
    wordle              = vf.load_environment("wordle")

    # Prepare each child env (normalize info + namespace extras)
    ifpt                = _prepare_env(ifpt, "ifpt")
    ifen                = _prepare_env(ifen, "ifen")
    hendrycksmath       = _prepare_env(hendrycksmath, "hendrycksmath")
    gsm8k               = _prepare_env(gsm8k, "gsm8k")
    logic_puzzle_solver = _prepare_env(logic_puzzle_solver, "logic_puzzle_solver")
    sudoku              = _prepare_env(sudoku, "sudoku")
    mini_sudoku         = _prepare_env(mini_sudoku, "mini_sudoku")
    wordle              = _prepare_env(sudoku, "wordle")

    # Now all datasets share only CORE_COLS as common names; everything else
    # is env-specific, so concatenate_datasets() won't hit type conflicts.
    return vf.EnvGroup(
        envs=[
            ifpt,
            ifen,
            hendrycksmath,
            gsm8k,
            logic_puzzle_solver,
            sudoku,
            mini_sudoku,
            #wordle
        ],
        env_names=[
            "ifpt",
            "ifen",
            "hendrycksmath",
            "gsm8k",
            "logic_puzzle_solver",
            "sudoku",
            "mini_sudoku",
            #"wordle"
        ],
        **kwargs,
    )
