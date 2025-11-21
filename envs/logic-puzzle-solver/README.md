# Logic Puzzle Solver Environment

### Overview
- **Environment ID**: `logic-puzzle-solver`
- **Short description**: An environment for solving logic puzzles with deductive reasoning
- **Tags**: logic, puzzles, reasoning, deduction, multi-turn

### Datasets
- **Primary dataset(s)**: Procedurally generated logic puzzles of varying difficulty
- **Source**: Generated at runtime using the `PuzzleGenerator` class
- **Split sizes**: Configurable via `num_puzzles` parameter (default: 100 puzzles)

### Task
- **Type**: Multi-turn conversational reasoning
- **Parser**: XMLParser with tags for `reasoning`, `deductions`, `question`, and `answer`
- **Rubric overview**: Correctness (50%), efficiency (20%), reasoning quality (20%), format compliance (10%)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval logic-puzzle-solver
```

Configure model and sampling:

```bash
uv run vf-eval logic-puzzle-solver -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"num_puzzles": 50, "difficulty_distribution": {"easy": 0.4, "medium": 0.4, "hard": 0.2}}'  
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_puzzles` | int | `100` | Number of puzzles to generate |
| `difficulty_distribution` | dict | `{"easy": 0.3, "medium": 0.4, "hard": 0.3}` | Distribution of puzzle difficulties |
| `max_turns` | int | `10` | Maximum number of turns allowed per puzzle |
| `min_turns` | int | `3` | Minimum number of turns before solution attempt is allowed |
| `seed` | int | `42` | Random seed for reproducibility |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `correctness_score` | Accuracy of the final solution (50% of reward) |
| `efficiency_score` | Solving the puzzle in fewer turns (20% of reward) |
| `reasoning_quality_score` | Quality and structure of reasoning (20% of reward) |
| `format_compliance_score` | Following the required response format (10% of reward) |

### Puzzle Structure

Each puzzle consists of:
- **Entities**: Individuals (e.g., Alice, Bob, Charlie)
- **Attributes**: Categories with values (e.g., colors, pets, drinks)
- **Clues**: Statements about relationships between entities and attributes

The task requires deducing which entity has which attribute values based on the given clues.

### Clue Types

The environment generates various types of clues:
1. **Direct clues**: "Alice has the red color."
2. **Negative clues**: "Bob does not have the cat."
3. **Relationship clues**: "The person with the cat has the blue color."
4. **Either/or clues**: "Either Alice or Bob has the dog (but not both)."
5. **Combination clues**: "Alice and Bob together have the cat and dog."

