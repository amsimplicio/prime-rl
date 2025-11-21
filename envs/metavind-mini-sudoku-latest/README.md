# mini-sudoku

### Overview

- **Environment ID**: `mini-sudoku`
- **Short description**: Solve 4x4 sudoku puzzle; rewards correctness, (optional) partial correctness, and format.
- **Tags**: puzzles, single-turn, sudoku, xml

### Datasets

- **Primary dataset(s)**: `metavind/mini-sudoku`
- **Source links**: <https://huggingface.co/datasets/metavind/mini-sudoku>
- **Split sizes**: 192 train examples, 96 test examples (per difficulty level)
- **Difficulty levels**:
  - `easy`: 1-4 empty cells
  - `medium`: 5-8 empty cells
  - `hard`: 9-12 empty cells
- **Note**: The dataset is filtered by the `difficulty` parameter during environment initialization

### Task

- **Type**: single-turn
- **Parser**: `XMLParser(["answer"])`
- **Rubric overview**: Exact solution match, (optional) partial correctness credit, and format check.
- **System prompt**: `Solve the following 4x4 sudoku puzzle by replacing all _ instances with the correct number such that each row, each column, and each of the four 2x2 blocks contains all numbers 1-4 exactly once. Provide your answer between <answer> and </answer> tags.`

<table>
  <tr>
    <th>Input Format</th>
    <th>Expected Answer</th>
  </tr>
  <tr>
    <td><pre>3 _ 4 _
4 _ 3 2
_ _ 1 4
_ _ _ 3</pre></td>
    <td><pre>3 2 4 1
4 1 3 2
2 3 1 4
1 4 2 3
</pre></td>
  </tr>
</table>

### Quickstart

Run an evaluation with default settings:

```bash
uv run vf-eval mini-sudoku
```

Configure model and sampling:

```bash
uv run vf-eval mini-sudoku \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 2048 -T 0.7 \
  -a '{"difficulty": "hard"}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Number of training examples to use (use -1 for all) |
| `num_eval_examples` | int | `-1` | Number of evaluation examples to use (use -1 for all) |
| `difficulty` | str | `"medium"` | Difficulty level to filter dataset by |
| `include_partial_credit` | bool | `True` | Whether to award partial credit for correct cells |

### Metrics

| Metric | Weight | Range | Meaning |
| ------ | ------ | ----- | ------- |
| Format reward | 0.1 | 0.0-0.8 | Adherence to XML format |
| Partial credit reward | 0.01 | 0.0-16.0 | Number of correct cells |
| Correct answer reward | 1.0 | 0.0-1.0 | Solution matches exactly |
