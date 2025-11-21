import random
from typing import List, Tuple
from .board import SudokuBoard


class SudokuGenerator:
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
    
    def generate_complete_solution(self) -> List[List[int]]:
        board = [[0 for _ in range(9)] for _ in range(9)]
        self._fill_board(board)
        return board
    
    def _fill_board(self, board: List[List[int]]) -> bool:
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    numbers = list(range(1, 10))
                    random.shuffle(numbers)
                    for num in numbers:
                        if self._is_valid_placement(board, i, j, num):
                            board[i][j] = num
                            if self._fill_board(board):
                                return True
                            board[i][j] = 0
                    return False
        return True
    
    def _is_valid_placement(self, board: List[List[int]], row: int, col: int, num: int) -> bool:
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
    
    def create_puzzle(self, difficulty: str = 'easy') -> Tuple[List[List[int]], List[List[int]]]:
        solution = self.generate_complete_solution()
        puzzle = [row[:] for row in solution]
        
        difficulty_settings = {
            'easy': {'min_clues': 36, 'max_clues': 46},
            'medium': {'min_clues': 28, 'max_clues': 35}, 
            'hard': {'min_clues': 22, 'max_clues': 27}
        }
        
        settings = difficulty_settings.get(difficulty, difficulty_settings['easy'])
        target_clues = random.randint(settings['min_clues'], settings['max_clues'])
        
        cells_to_remove = 81 - target_clues
        cells = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(cells)
        
        removed = 0
        max_attempts = min(cells_to_remove + 10, len(cells))
        
        for idx, (row, col) in enumerate(cells):
            if removed >= cells_to_remove or idx >= max_attempts:
                break
                
            backup = puzzle[row][col]
            puzzle[row][col] = 0
            
            if self._is_still_solvable(puzzle):
                removed += 1
            else:
                puzzle[row][col] = backup
        
        return puzzle, solution
    
    def _is_still_solvable(self, puzzle: List[List[int]]) -> bool:
        test_board = [row[:] for row in puzzle]
        return self._solve_deterministic(test_board)
    
    def _solve_deterministic(self, board: List[List[int]]) -> bool:
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if self._is_valid_placement(board, i, j, num):
                            board[i][j] = num
                            if self._solve_deterministic(board):
                                return True
                            board[i][j] = 0
                    return False
        return True
    
    def _has_unique_solution_fast(self, puzzle: List[List[int]]) -> bool:
        solutions = []
        self._find_all_solutions(puzzle, solutions, max_solutions=2)
        return len(solutions) == 1
    
    def _find_all_solutions(self, board: List[List[int]], solutions: List, max_solutions: int = 2):
        if len(solutions) >= max_solutions:
            return
        
        empty_cell = self._find_empty_cell(board)
        if not empty_cell:
            solutions.append([row[:] for row in board])
            return
        
        row, col = empty_cell
        for num in range(1, 10):
            if self._is_valid_placement(board, row, col, num):
                board[row][col] = num
                self._find_all_solutions(board, solutions, max_solutions)
                board[row][col] = 0
    
    def _find_empty_cell(self, board: List[List[int]]):
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None
    
    def _validate_puzzle_quality(self, puzzle: List[List[int]], solution: List[List[int]]) -> bool:
        test_board = [row[:] for row in puzzle]
        if not self._solve_deterministic(test_board):
            return False
        
        for i in range(9):
            for j in range(9):
                if test_board[i][j] != solution[i][j]:
                    return False
        
        empty_cells = sum(1 for row in puzzle for cell in row if cell == 0)
        if empty_cells < 20 or empty_cells > 60:
            return False
        
        return True


def generate_puzzle(difficulty: str = 'easy', seed: int = None) -> Tuple[List[List[int]], List[List[int]]]:
    generator = SudokuGenerator(seed)
    return generator.create_puzzle(difficulty)


def validate_difficulty(puzzle: List[List[int]]) -> str:
    clue_count = sum(1 for row in puzzle for cell in row if cell != 0)
    
    if clue_count >= 36:
        return 'easy'
    elif clue_count >= 28:
        return 'medium'
    else:
        return 'hard'
