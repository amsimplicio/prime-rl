import random
from typing import List
from datasets import Dataset
from .board import SudokuBoard
from .generator import generate_puzzle


def solve_sudoku(board: List[List[int]]) -> List[List[int]]:
    def is_valid(board, row, col, num):
        for x in range(9):
            if board[row][x] == num or board[x][col] == num:
                return False
        
        start_row = row - row % 3
        start_col = col - col % 3
        for i in range(3):
            for j in range(3):
                if board[i + start_row][j + start_col] == num:
                    return False
        return True
    
    def solve(board):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if solve(board):
                                return True
                            board[i][j] = 0
                    return False
        return True
    
    solution = [row[:] for row in board]
    solve(solution)
    return solution


def validate_puzzle(puzzle: List[List[int]]) -> bool:
    try:
        solution = solve_sudoku(puzzle)
        board = SudokuBoard(solution)
        return board.is_solved()
    except:
        return False


def load_sudoku_dataset(
    max_examples: int = 100,
    seed: int = 42,
    difficulty: str = 'easy'
) -> Dataset:
    if seed is not None:
        random.seed(seed)
    
    dataset_rows = []
    
    for i in range(max_examples):
        puzzle_seed = seed + i if seed is not None else None
        puzzle, solution = generate_puzzle(difficulty=difficulty, seed=puzzle_seed)
        
        board = SudokuBoard(puzzle)
        question = f"""Sudoku:

{board.to_string()}

Regras:
- Preenche cada célula vazia (.) com um número de 1-9
- Cada linha deve conter todos os números 1-9 exatamente uma vez
- Cada coluna deve conter todos os números 1-9 exatamente uma vez  
- Cada caixa 3x3 deve conter todos os números 1-9 exatamente uma vez

Faz uma jogada de cada vez usando o formato: <move>A1=5</move>
Onde A1 representa a linha A, coluna 1, e 5 é o número a colocar."""
        
        answer = f"Complete solution available (puzzle is solvable)"
        answer = f"Solução completa disponível (é possível resolver o puzzle)"
        
        stats = board.get_progress_stats()
        
        dataset_rows.append({
            "question": question,
            "answer": answer,
            "info": {
                "puzzle": puzzle,
                "solution": solution,
                "difficulty": difficulty,
                "initial_clues": stats['initial_clues'],
                "total_moves_needed": stats['empty_cells']
            }
        })
    
    return Dataset.from_list(dataset_rows)