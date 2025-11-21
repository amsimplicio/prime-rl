import re
from typing import Tuple, Optional


def extract_move(text: str) -> str:
    move_match = re.search(r'<move>(.*?)</move>', text, re.DOTALL | re.IGNORECASE)
    if move_match:
        return move_match.group(1).strip()
    return ""


def parse_move_string(move_str: str) -> Optional[Tuple[int, int, int]]:
    if not move_str:
        return None
    
    move_str = move_str.strip().upper()
    
    match = re.match(r'^([A-I])([1-9])=([0-9])$', move_str)
    if not match:
        return None
    
    row_letter, col_str, num_str = match.groups()
    
    row_index = ord(row_letter) - ord('A')
    col_index = int(col_str) - 1
    number = int(num_str)
    
    if not (0 <= row_index < 9 and 0 <= col_index < 9 and 0 <= number <= 9):
        return None
    
    return (row_index, col_index, number)


def format_move(row: int, col: int, num: int) -> str:
    row_letter = chr(ord('A') + row)
    col_num = col + 1
    return f"{row_letter}{col_num}={num}"


def parse_coordinate(coord_str: str) -> Optional[Tuple[int, int]]:
    if not coord_str or len(coord_str) != 2:
        return None
    
    coord_str = coord_str.upper()
    row_letter = coord_str[0]
    col_char = coord_str[1]
    
    if not ('A' <= row_letter <= 'I' and '1' <= col_char <= '9'):
        return None
    
    row_index = ord(row_letter) - ord('A')
    col_index = int(col_char) - 1
    
    return (row_index, col_index)