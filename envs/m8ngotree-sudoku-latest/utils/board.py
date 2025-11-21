from typing import List, Tuple, Set, Optional
import copy


class SudokuBoard:
    
    def __init__(self, initial_board: List[List[int]] = None):
        if initial_board is None:
            self.board = [[0 for _ in range(9)] for _ in range(9)]
        else:
            self.board = copy.deepcopy(initial_board)
        
        self.initial_board = copy.deepcopy(self.board)
        self.move_history = []
        self.constraint_violations = 0
    
    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        if not (0 <= row < 9 and 0 <= col < 9 and 1 <= num <= 9):
            return False
        
        if self.board[row][col] != 0:
            return False
        
        return (not self._in_row(row, num) and 
                not self._in_col(col, num) and 
                not self._in_box(row, col, num))
    
    def make_move(self, row: int, col: int, num: int) -> bool:
        if not (0 <= row < 9 and 0 <= col < 9):
            return False
        
        if num == 0:
            if self.board[row][col] != 0:
                old_num = self.board[row][col]
                self.board[row][col] = 0
                self.move_history.append((row, col, 0))
                return True
            else:
                return False
        
        if self.is_valid_move(row, col, num):
            self.board[row][col] = num
            self.move_history.append((row, col, num))
            return True
        else:
            self.constraint_violations += 1
            return False
    
    def undo_last_move(self) -> bool:
        if not self.move_history:
            return False
        
        row, col, num = self.move_history.pop()
        self.board[row][col] = 0
        return True
    
    def is_complete(self) -> bool:
        for row in self.board:
            if 0 in row:
                return False
        return True
    
    def is_solved(self) -> bool:
        if not self.is_complete():
            return False
        
        for i in range(9):
            for j in range(9):
                num = self.board[i][j]
                self.board[i][j] = 0
                if not self.is_valid_move(i, j, num):
                    self.board[i][j] = num
                    return False
                self.board[i][j] = num
        return True
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        empty = []
        for i in range(9):
            for j in range(9):
                if self.board[i][j] == 0:
                    empty.append((i, j))
        return empty
    
    def get_possible_values(self, row: int, col: int) -> Set[int]:
        if self.board[row][col] != 0:
            return set()
        
        possible = set(range(1, 10))
        
        for num in range(1, 10):
            if (self._in_row(row, num) or 
                self._in_col(col, num) or 
                self._in_box(row, col, num)):
                possible.discard(num)
        
        return possible
    
    def _in_row(self, row: int, num: int) -> bool:
        return num in self.board[row]
    
    def _in_col(self, col: int, num: int) -> bool:
        return num in [self.board[row][col] for row in range(9)]
    
    def _in_box(self, row: int, col: int, num: int) -> bool:
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        
        for i in range(box_row, box_row + 3):
            for j in range(box_col, box_col + 3):
                if self.board[i][j] == num:
                    return True
        return False
    
    def to_string(self) -> str:
        result = "  1 2 3 | 4 5 6 | 7 8 9\n"
        rows = "ABCDEFGHI"
        
        for i, row in enumerate(self.board):
            if i in [3, 6]:
                result += "  ------+-------+------\n"
            
            result += f"{rows[i]} "
            for j, cell in enumerate(row):
                if j in [3, 6]:
                    result += "| "
                
                if cell == 0:
                    result += ". "
                else:
                    result += f"{cell} "
            result += "\n"
        
        return result
    
    def get_progress_stats(self) -> dict:
        total_cells = 81
        filled_cells = sum(1 for row in self.board for cell in row if cell != 0)
        initial_filled = sum(1 for row in self.initial_board for cell in row if cell != 0)
        player_moves = filled_cells - initial_filled
        
        return {
            'total_cells': total_cells,
            'filled_cells': filled_cells,
            'empty_cells': total_cells - filled_cells,
            'initial_clues': initial_filled,
            'player_moves': player_moves,
            'completion_percentage': (filled_cells / total_cells) * 100,
            'moves_made': len(self.move_history),
            'constraint_violations': self.constraint_violations
        }