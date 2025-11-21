import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer
from datasets import load_dataset


def load_environment(num_train_examples: int = -1, num_eval_examples: int = -1):
    """
    Single-turn MedMCQA environment using the `medmcqa` dataset from HF.

    - Formats each example as chat messages: a system prompt requiring CoT inside
      <think>...</think> and a final answer letter inside \boxed{A|B|C|D}.
    - Rewards exact match on the final letter parsed from \boxed{...}.
    """

    # Load dataset
    ds_all = load_dataset("medmcqa")
    dataset = ds_all["train"]

    # Pick eval split
    if "test" in ds_all:
        eval_dataset = ds_all["test"]
    elif "validation" in ds_all:
        eval_dataset = ds_all["validation"]
    else:
        # Create a holdout from train if needed
        if num_eval_examples != -1:
            eval_size = max(1, min(num_eval_examples, len(dataset) - 1 if len(dataset) > 1 else 1))
        else:
            eval_size = max(1, min(200, max(1, len(dataset) // 10)))
        split = dataset.train_test_split(test_size=eval_size, seed=42, shuffle=True)
        dataset = split["train"]
        eval_dataset = split["test"]

    if num_train_examples != -1:
        dataset = dataset.select(range(min(num_train_examples, len(dataset))))
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(min(num_eval_examples, len(eval_dataset))))

    system_prompt = """
    You are a medical QA assistant. Think step-by-step inside <think>...</think> to arrive at the best answer.

    Then, output your final choice inside \\boxed{...} using exactly one token from: A, B, C, D.
    Do not include option text in the box; only the letter.
    """

    def _doc_to_text(doc) -> str:
        # Provided helper to construct the prompt text
        choices = [doc.get("opa", ""), doc.get("opb", ""), doc.get("opc", ""), doc.get("opd", "")]
        option_choices = {
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
        }

        prompt = "Question: " + (doc.get("question") or "") + "\nChoices:\n"
        for choice, option in option_choices.items():
            prompt += f"{choice}. {option}\n"
        prompt += "Answer:"
        return prompt

    def _normalize_opt_text(s: str) -> str:
        import re

        t = (s or "").strip()
        # Remove leading labels like "A.", "B)", etc.
        t = re.sub(r"^[A-Da-d]\s*[\.)\:\-\]]\s*", "", t)
        t = re.sub(r"\s+", " ", t)
        return t.lower()

    def _letter_to_idx(letter: str):
        if not isinstance(letter, str):
            return None
        u = letter.strip().upper()
        if u in {"A", "B", "C", "D"}:
            return ["A", "B", "C", "D"].index(u)
        return None

    def _infer_answer_idx(example, options):
        # Try explicit letter in 'cop'
        cop = example.get("cop")
        idx = None
        if cop is not None:
            # letter form
            idx = _letter_to_idx(str(cop))
            if idx is not None:
                return idx
            # numeric string/int 0-3 or 1-4
            try:
                iv = int(str(cop).strip())
                if 0 <= iv < len(options):
                    return iv
                if 1 <= iv <= len(options):
                    return iv - 1
            except Exception:
                pass
            # text of the correct option
            try:
                norm_opts = [_normalize_opt_text(o) for o in options]
                s_norm = _normalize_opt_text(str(cop))
                if s_norm in norm_opts:
                    return norm_opts.index(s_norm)
            except Exception:
                pass

        # Sometimes datasets store the full correct option string in 'exp' or similar; try matching
        for key in ["answer", "label", "target"]:
            if key in example:
                v = example.get(key)
                # Try letter
                idx = _letter_to_idx(str(v))
                if idx is not None:
                    return idx
                # Try numeric
                try:
                    iv = int(str(v).strip())
                    if 0 <= iv < len(options):
                        return iv
                    if 1 <= iv <= len(options):
                        return iv - 1
                except Exception:
                    pass
                # Try text match
                try:
                    norm_opts = [_normalize_opt_text(o) for o in options]
                    s_norm = _normalize_opt_text(str(v))
                    if s_norm in norm_opts:
                        return norm_opts.index(s_norm)
                except Exception:
                    pass
        return None

    def _format_example(example):
        options = [
            example.get("opa", ""),
            example.get("opb", ""),
            example.get("opc", ""),
            example.get("opd", ""),
        ]
        # Normalize to strings
        options = ["" if o is None else str(o) for o in options]

        # Build prompt with Choices listed in a deterministic random label order (display only)
        import hashlib
        import random as _random

        q = (example.get("question") or "").strip()
        label_to_opt = {"A": options[0], "B": options[1], "C": options[2], "D": options[3]}
        labels = ["A", "B", "C", "D"]
        # Seed per-example for reproducible but shuffled display order
        id_str = str(example.get("id") if example.get("id") is not None else q)
        seed = int(hashlib.md5(id_str.encode("utf-8")).hexdigest()[:8], 16)
        rnd = _random.Random(seed)
        rnd.shuffle(labels)

        prompt_text = "Question: " + q + "\nChoices:\n"
        for lab in labels:
            prompt_text += f"{lab}. {label_to_opt[lab]}\n"
        prompt_text += "Answer:"
        # Add a trailing instruction to be safe
        user_content = f"{prompt_text}\nAnswer with the letter only inside \\boxed{{...}} (A, B, C, or D)."

        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_content},
        ]

        # Determine correct answer index/letter
        ans_idx = _infer_answer_idx(example, options)
        if isinstance(ans_idx, int) and 0 <= ans_idx < 4:
            ans_letter = ["A", "B", "C", "D"][ans_idx]
        else:
            ans_letter = None
            ans_idx = -1

        example["prompt"] = messages
        example["options"] = options
        example["answer_idx"] = ans_idx
        example["answer_letter"] = ans_letter if ans_letter else ""
        example["answer"] = ans_letter if ans_letter else ""
        return example

    dataset = dataset.map(_format_example)
    eval_dataset = eval_dataset.map(_format_example)

    # Filter to valid examples
    def _keep_example(example):
        opts = example.get("options") or []
        idx = example.get("answer_idx")
        if not isinstance(opts, list) or len(opts) < 2:
            return False
        try:
            return isinstance(idx, int) and 0 <= idx < len(opts)
        except Exception:
            return False

    dataset = dataset.filter(_keep_example)
    eval_dataset = eval_dataset.filter(_keep_example)

    parser = vf.ThinkParser(extract_fn=extract_boxed_answer)

    def _completion_to_text(completion):
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            if completion and isinstance(completion[0], dict):
                for msg in reversed(completion):
                    content = msg.get("content")
                    if content:
                        if isinstance(content, list):
                            try:
                                content = " ".join(
                                    (
                                        part.get("text", "") if isinstance(part, dict) else str(part)
                                    )
                                    for part in content
                                )
                            except Exception:
                                content = str(content)
                        return str(content)
                return str(completion[-1])
            return "\n".join(str(x) for x in completion)
        if isinstance(completion, dict):
            content = completion.get("content") or completion.get("text")
            if isinstance(content, list):
                try:
                    content = " ".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                except Exception:
                    content = str(content)
            if content is not None:
                return str(content)
        return str(completion)

    # Reward: 1.0 if the boxed letter equals the expected letter; accept numeric 0-3 too
    def correct_letter_reward_func(completion, answer=None, answer_letter=None, answer_idx=None, **kwargs):
        completion_text = _completion_to_text(completion)
        parsed = (parser.parse_answer(completion_text) or "").strip()

        # Expected
        expected_letter = None
        if isinstance(answer_letter, str) and answer_letter.strip().upper() in {"A", "B", "C", "D"}:
            expected_letter = answer_letter.strip().upper()
        elif isinstance(answer, str) and answer.strip().upper() in {"A", "B", "C", "D"}:
            expected_letter = answer.strip().upper()
        elif isinstance(answer_idx, int) and 0 <= answer_idx <= 3:
            expected_letter = ["A", "B", "C", "D"][answer_idx]

        # Predicted
        predicted_letter = None
        if parsed.upper() in {"A", "B", "C", "D"}:
            predicted_letter = parsed.upper()
        else:
            try:
                iv = int(parsed)
                if 0 <= iv <= 3:
                    predicted_letter = ["A", "B", "C", "D"][iv]
            except Exception:
                pass

        if expected_letter is None or predicted_letter is None:
            return 0.0
        return 1.0 if predicted_letter == expected_letter else 0.0

    rubric = vf.Rubric(
        parser=parser,
        funcs=[
            correct_letter_reward_func,
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.0],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
