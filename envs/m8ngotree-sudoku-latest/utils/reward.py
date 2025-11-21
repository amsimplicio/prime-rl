from typing import Dict, Any
from .board import SudokuBoard


def calculate_move_reward(
    board: SudokuBoard, 
    move_valid: bool, 
    move_info: Dict[str, Any] = None
) -> float:
    if move_info is None:
        move_info = {}
    
    if not move_valid:
        return -0.2
    
    is_backtrack = move_info.get('is_backtrack', False)
    
    if is_backtrack:
        return -0.1
    
    reward = 0.1
    
    if board.is_solved():
        reward += 2.0
        return reward
    
    stats = board.get_progress_stats()
    
    progress_bonus = stats['completion_percentage'] / 1000
    reward += progress_bonus
    
    if 'possibilities_before' in move_info and 'possibilities_after' in move_info:
        possibilities_reduced = move_info['possibilities_before'] - move_info['possibilities_after']
        if possibilities_reduced > 0:
            reward += min(0.05, possibilities_reduced * 0.01)
    
    if move_info.get('completed_constraint', False):
        reward += 0.5
    
    return reward


def calculate_final_reward(board: SudokuBoard, max_turns: int, turns_used: int) -> float:
    base_reward = 0.0
    
    if board.is_solved():
        base_reward = 5.0
        
        if max_turns > 0:
            efficiency_ratio = 1.0 - (turns_used / max_turns)
            efficiency_bonus = efficiency_ratio * 1.0
            base_reward += efficiency_bonus
    
    elif board.is_complete():
        base_reward = 1.0
    
    else:
        stats = board.get_progress_stats()
        completion_reward = (stats['completion_percentage'] / 100) * 2.0
        base_reward = completion_reward
    
    violation_penalty = board.constraint_violations * 0.1
    base_reward -= violation_penalty
    
    return max(0.0, base_reward)


def get_move_feedback(
    board: SudokuBoard, 
    move_valid: bool, 
    row: int = None, 
    col: int = None, 
    num: int = None
) -> str:
    if move_valid:
        if num == 0:
            stats = board.get_progress_stats()
            empty_cells = stats['empty_cells']
            return f"Retrocedido! Célula limpa. {empty_cells} células restantes."
        elif board.is_solved():
            return "Parabéns! Resolveste o puzzle!"
        elif board.is_complete():
            return "O tabuleiro está completo mas contém erros. Verifica a tua solução."
        else:
            stats = board.get_progress_stats()
            empty_cells = stats['empty_cells']
            return f"Jogada válida! {empty_cells} células restantes."
    
    else:
        if row is not None and col is not None and num is not None:
            row_letter = chr(ord('A') + row)
            col_num = col + 1
            
            if num == 0 and board.board[row][col] == 0:
                return f"A célula {row_letter}{col_num} já está vazia!"
            elif num != 0 and board.board[row][col] != 0:
                return f"A célula {row_letter}{col_num} já está preenchida!"
            elif num != 0 and board._in_row(row, num):
                return f"O número {num} já existe na linha {row_letter}!"
            elif num != 0 and board._in_col(col, num):
                return f"O número {num} já existe na coluna {col_num}!"
            elif num != 0 and board._in_box(row, col, num):
                box_row = (row // 3) + 1
                box_col = (col // 3) + 1
                return f"O número {num} já existe na caixa {box_row}{box_col}!"
        
        return "Jogada inválida! Verifica as restrições do Sudoku."