import numpy as np
from datasets import load_dataset

import verifiers as vf

### system prompt
SYSTEM_PROMPT = """\
Resolve o seguinte puzzle de sudoku 4x4 substituindo todas as instâncias de _ pelo número correto de forma a que \
cada linha, cada coluna e cada um dos quatro blocos 2x2 contenha todos os números 1-4 exatamente uma vez. \
Fornece a tua resposta entre as tags <answer> e </answer>."""

### helper functions
def board_from_string(board_str: str) -> np.ndarray:
    """Convert a string representation to a numpy board"""
    rows = board_str.strip().split("\n")
    return np.array([[0 if cell == "_" else int(cell) for cell in row.split()] 
                     for row in rows])


def is_valid_board(board: np.ndarray) -> bool:
    """Check if board meets game rules (4x4 shape, values between 1 and 4)."""
    return board.shape == (4, 4) and np.all((1 <= board) & (board <= 4)).item()


def parse_guess(guess: str | None) -> np.ndarray | None:
    """Parse and validate a guess."""
    if guess is None:
        return None
    try:
        guess_board = board_from_string(guess)
        return guess_board if is_valid_board(guess_board) else None 
    except Exception:
        return None


### reward functions
def partial_credit_reward_func(parser, completion, answer) -> float:
    """Reward +1.0 partial credit for each correct cell in guess."""
    parsed_guess = parser.parse_answer(completion)
    guess_board = parse_guess(parsed_guess)
    if guess_board is None:
        return 0.0

    answer_board = board_from_string(answer)
    return float(np.sum(np.equal(guess_board, answer_board)))

def correct_answer_reward_func(parser, completion, answer) -> float:
    """Return +1.0 when the guess matches the ground-truth exactly."""
    parsed_guess = parser.parse_answer(completion)
    guess_board = parse_guess(parsed_guess)
    if guess_board is None:
        return 0.0

    answer_board = board_from_string(answer)
    return float(np.array_equal(guess_board, answer_board))


### environment loader
def load_environment(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    difficulty: str = "medium",
    include_partial_credit: bool = True,
) -> vf.Environment:
    dataset_dict = load_dataset("metavind/mini-sudoku")
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]

    train_dataset = train_dataset.filter(lambda x: x["difficulty"] == difficulty)  # type: ignore
    eval_dataset = eval_dataset.filter(lambda x: x["difficulty"] == difficulty)  # type: ignore

    if num_train_examples != -1:
        train_dataset = train_dataset.select(range(num_train_examples))  # type: ignore
    if num_eval_examples != -1:
        eval_dataset = eval_dataset.select(range(num_eval_examples))  # type: ignore

    def attach_info(sample: dict) -> dict:
        info = {"difficulty": sample["difficulty"], "empty_cells": sample["empty_cells"]}
        return {"info": info}

    train_dataset = train_dataset.map(attach_info)  # type: ignore
    eval_dataset = eval_dataset.map(attach_info)  # type: ignore

    parser = vf.XMLParser(fields=["answer"], answer_field="answer")

    rubric = vf.Rubric(parser=parser)
    rubric.add_reward_func(parser.get_format_reward_func(), weight=0.1)
    if include_partial_credit:
        rubric.add_reward_func(partial_credit_reward_func, weight=0.01)
    rubric.add_reward_func(correct_answer_reward_func)
    
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )

    return vf_env
