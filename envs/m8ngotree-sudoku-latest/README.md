# sudoku

### Overview
- **Environment ID**: `sudoku`
- **Short description**: Multi-turn Sudoku puzzle solving environment with algorithmic generation and difficulty levels
- **Tags**: multi-turn, reasoning, constraints, logic, train, eval, think

### Datasets
- **Primary dataset(s)**: Algorithmically generated Sudoku puzzles with configurable difficulty
- **Source links**: Dynamic puzzle generation with unique solution validation
- **Split sizes**: Configurable via `max_examples` parameter (default: 100 puzzles)

### Task
- **Type**: multi-turn (interactive solving)
- **Parser**: `ThinkParser` (with thinking) or `Parser` (direct) with move extraction
- **Rubric overview**: Progressive rewards for valid moves, completion bonuses, and efficiency scoring

### Game Rules
The Sudoku environment presents models with:
- **9x9 Grid**: Standard Sudoku board with algorithmically generated clues
- **Difficulty Levels**: Easy (36-46 clues), Medium (28-35 clues), Hard (22-27 clues)
- **Constraints**: Each row, column, and 3x3 box must contain digits 1-9 exactly once
- **Single Move**: Make one move per turn in format `<move>A1=5</move>`
- **Feedback**: Receive validation and board state after each move
- **Goal**: Complete the puzzle with all constraints satisfied

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval sudoku
```

Configure model and sampling:

```bash
uv run vf-eval sudoku \
  -m gpt-4o-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"max_examples": 50, "use_think": true, "max_turns": 30}'
```

Test different configurations:

```bash
# Direct reasoning (no <think> tags)
uv run vf-eval sudoku -a '{"use_think": false}'

# Medium difficulty puzzles
uv run vf-eval sudoku -a '{"difficulty": "medium"}'

# Hard difficulty with more turns
uv run vf-eval sudoku -a '{"difficulty": "hard", "max_turns": 75}'

# Different random seed
uv run vf-eval sudoku -a '{"seed": 123}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_examples` | int | `100` | Number of Sudoku puzzles to include in dataset |
| `use_think` | bool | `True` | Use thinking mode with `<think>` tags (`ThinkParser`) or direct reasoning (`Parser`) |
| `max_turns` | int | `50` | Maximum number of moves allowed per puzzle |
| `seed` | int | `42` | Random seed for reproducible puzzle selection |
| `difficulty` | str | `"easy"` | Puzzle difficulty level: "easy" (36-46 clues), "medium" (28-35 clues), "hard" (22-27 clues) |

### Scoring System

**Progressive Reward System (Cumulative)**

Per-move rewards:
- **+0.1 points**: Each valid move
- **-0.1 points**: Backtracking moves (clearing cells with 0)
- **-0.2 points**: Invalid moves (constraint violations)

Final rewards:
- **+5.0 points**: Successfully solving the puzzle
- **+1.0 points**: Efficiency bonus based on turns used vs. max turns
- **+1.0 points**: Completing puzzle (even if invalid)
- **+2.0 points**: Partial completion based on percentage filled
- **-0.1 points**: Penalty per constraint violation

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `sudoku_reward` | Cumulative reward based on moves and completion |

### Example Interaction

**Initial Puzzle:**
```
Sudoku Puzzle:

  1 2 3 | 4 5 6 | 7 8 9
A 5 3 . | . 7 . | . . .
B 6 . . | 1 9 5 | . . .
C . 9 8 | . . . | . 6 .
  ------+-------+------
D 8 . . | . 6 . | . . 3
E 4 . . | 8 . 3 | . . 1
F 7 . . | . 2 . | . . 6
  ------+-------+------
G . 6 . | . . . | 2 8 .
H . . . | 4 1 9 | . . 5
I . . . | . 8 . | . 7 9

Rules:
- Fill each empty cell (.) with a number from 1-9
- Each row must contain all numbers 1-9 exactly once
- Each column must contain all numbers 1-9 exactly once  
- Each 3x3 box must contain all numbers 1-9 exactly once

Make one move at a time using the format: <move>A1=5</move>
```

**Expected Response Format:**
```
<think>
Looking at row A, column 3 (A3). 
Row A already has: 5, 3, 7
Row A is missing: 1, 2, 4, 6, 8, 9

Column 3 already has: 8
Column 3 is missing: 1, 2, 3, 4, 5, 6, 7, 9

Box 1 (top-left) already has: 5, 3, 6, 9, 8
Box 1 is missing: 1, 2, 4, 7

For A3, the possible values are the intersection: 1, 2, 4
Need to check other constraints to narrow it down...
Looking at other cells, A3 can only be 4.
</think>

<move>A3=4</move>
```

**Environment Response:**
```
✅ Valid move! 47 cells remaining.

Current Board:
  1 2 3 | 4 5 6 | 7 8 9
A 5 3 4 | . 7 . | . . .
B 6 . . | 1 9 5 | . . .
C . 9 8 | . . . | . 6 .
  ------+-------+------
D 8 . . | . 6 . | . . 3
E 4 . . | 8 . 3 | . . 1
F 7 . . | . 2 . | . . 6
  ------+-------+------
G . 6 . | . . . | 2 8 .
H . . . | 4 1 9 | . . 5
I . . . | . 8 . | . 7 9

Progress: 34/81 cells filled (42.0%)
Turn: 1/50

What's your next move?
```

### Move Format

**Required Format**: `<move>RowColumn=Number</move>`

- **Row**: A-I (A=top row, I=bottom row)
- **Column**: 1-9 (1=leftmost, 9=rightmost)  
- **Number**: 1-9 (digit to place)

**Valid Examples**:
- `<move>A1=5</move>` - Place 5 in top-left corner
- `<move>E5=7</move>` - Place 7 in center cell
- `<move>I9=3</move>` - Place 3 in bottom-right corner
- `<move>A1=0</move>` - Clear cell A1 (backtrack)

**Invalid Examples**:
- `<move>A1 = 5</move>` - Extra spaces
- `<move>A10=5</move>` - Column 10 doesn't exist
- `<move>J1=5</move>` - Row J doesn't exist

### Game Flow

1. **Initialization**: Present puzzle with clues and empty cells
2. **Turn Loop**: 
   - Model analyzes board and makes one move
   - Environment validates move and updates board
   - Environment provides feedback and updated board state
   - Continue until solved, max turns reached, or error
3. **Completion**: Calculate final reward based on success and efficiency

### Features

- **Algorithmic Generation**: Dynamic puzzle creation with unique solution validation
- **Multiple Difficulty Levels**: Easy, medium, and hard puzzles with appropriate clue counts
- **Backtracking Support**: Models can clear cells and try different approaches
- **Penalty-Based Learning**: Backtracking incurs small penalty to encourage efficiency
- **Constraint Validation**: Real-time checking of Sudoku rules
- **Progressive Feedback**: Detailed messages for valid/invalid moves including backtracks
- **Scalable Dataset**: Generate unlimited unique puzzles for training
- **Efficiency Scoring**: Rewards for solving in fewer moves
- **Error Recovery**: Continues game after invalid moves with feedback