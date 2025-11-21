import verifiers as vf
from typing import Tuple, Dict, Any
from utils import (
    SudokuBoard,
    load_sudoku_dataset,
    extract_move,
    parse_move_string,
    calculate_move_reward,
    calculate_final_reward,
    get_move_feedback,
    SUDOKU_SYSTEM_PROMPT,
    THINK_SUDOKU_SYSTEM_PROMPT
)


class SudokuEnv(vf.MultiTurnEnv):
    
    def __init__(
        self,
        dataset=None,
        eval_dataset=None,
        system_prompt: str = None,
        use_think: bool = True,
        max_turns: int = 50,
        **kwargs
    ):
        if eval_dataset is None and dataset is not None:
            eval_dataset = dataset
        
        super().__init__(
            max_turns=max_turns, 
            dataset=dataset,
            eval_dataset=eval_dataset,
            **kwargs
        )
        self.use_think = use_think
        
        if system_prompt is None:
            self.system_prompt = THINK_SUDOKU_SYSTEM_PROMPT if use_think else SUDOKU_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt
        
        if use_think:
            self.parser = vf.ThinkParser(extract_fn=extract_move)
        else:
            self.parser = vf.Parser(extract_fn=extract_move)
        
        self.rubric = vf.Rubric(
            funcs=[self._calculate_reward],
            weights=[1.0],
            parser=self.parser
        )
    
    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        if self.eval_dataset and len(self.eval_dataset) > 0:
            example_idx = state.get('example_idx', 0) % len(self.eval_dataset)
            example = self.eval_dataset[example_idx]
            puzzle = example['info']['puzzle']
        else:
            puzzle = [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9]
            ]
        
        board = SudokuBoard(puzzle)
        
        state.update({
            'board': board,
            'turn_count': 0,
            'total_reward': 0.0,
            'game_complete': False,
            'last_move_valid': True,
            'feedback_message': "A começar um novo jogo de sudoku!"
        })
        
        return state
    
    async def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs) -> bool:
        board = state.get('board')
        turn_count = state.get('turn_count', 0)
        
        if not board:
            return False
        
        return (board.is_solved() or 
                turn_count >= self.max_turns or 
                state.get('game_complete', False))
    
    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> Tuple[vf.Messages, vf.State]:
        board = state['board']
        turn_count = state.get('turn_count', 0)
        
        if messages and messages[-1]['role'] == 'assistant':
            move_text = messages[-1]['content']
            
            move_str = extract_move(move_text)
            parsed_move = parse_move_string(move_str)
            
            if parsed_move:
                row, col, num = parsed_move
                
                move_valid = board.make_move(row, col, num)
                
                move_info = {'is_backtrack': num == 0}
                move_reward = calculate_move_reward(board, move_valid, move_info)
                
                feedback = get_move_feedback(board, move_valid, row, col, num)
                
                state.update({
                    'turn_count': turn_count + 1,
                    'total_reward': state.get('total_reward', 0.0) + move_reward,
                    'last_move_valid': move_valid,
                    'feedback_message': feedback
                })
                
            else:
                feedback = "Jogada inválida! Use: <move>A1=5</move>"
                state.update({
                    'turn_count': turn_count + 1,
                    'last_move_valid': False,
                    'feedback_message': feedback
                })
        
        if board.is_solved():
            state['game_complete'] = True
            final_reward = calculate_final_reward(board, self.max_turns, state['turn_count'])
            state['total_reward'] = final_reward
        
        board_display = board.to_string()
        stats = board.get_progress_stats()
        feedback_msg = state.get('feedback_message', '')
        
        response_content = f"""{feedback_msg}

Tabuleiro atual:
{board_display}

Progresso: {stats['filled_cells']}/81 células preenchidas ({stats['completion_percentage']:.1f}%)
Turno: {state['turn_count']}/{self.max_turns}

Qual é a tua próxima jogada?"""
        
        response_messages = [{"role": "user", "content": response_content}]
        
        return response_messages, state
    
    def _calculate_reward(self, parser, completion, info=None, **kwargs) -> float:
        state = kwargs.get('state', {})
        return state.get('total_reward', 0.0)


def load_environment(
    max_examples: int = 10000,
    use_think: bool = True,
    max_turns: int = 50,
    seed: int = 42,
    difficulty: str = 'easy',
    **kwargs
) -> SudokuEnv:
    
    dataset = load_sudoku_dataset(
        max_examples=max_examples,
        seed=seed,
        difficulty=difficulty
    )
    
    env = SudokuEnv(
        dataset=dataset,
        use_think=use_think,
        max_turns=max_turns,
        **kwargs
    )
    
    return env