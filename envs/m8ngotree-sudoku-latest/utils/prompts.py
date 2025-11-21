SUDOKU_SYSTEM_PROMPT = """Estás a resolver um puzzle de Sudoku passo a passo.

REGRAS:
- Preenche cada célula vazia (.) com um número de 1-9
- Cada linha deve conter todos os números 1-9 exatamente uma vez
- Cada coluna deve conter todos os números 1-9 exatamente uma vez  
- Cada caixa 3x3 deve conter todos os números 1-9 exatamente uma vez
- Faz UMA jogada de cada vez

FORMATO DA JOGADA:
DEVES sempre fornecer exatamente uma jogada neste formato: <move>A1=5</move>
- A-I representa as linhas (A=linha superior, I=linha inferior)
- 1-9 representa as colunas (1=mais à esquerda, 9=mais à direita)
- O número depois de = é o que queres colocar

ESTRATÉGIA:
- Procura células com apenas um número possível (singles isolados)
- Verifica as restrições de linha, coluna e caixa
- Faz deduções lógicas passo a passo
- Se ficares bloqueado, podes retroceder alterando uma célula previamente preenchida para 0: <move>A1=0</move>
- Depois de retroceder, tenta um número diferente nessa célula ou noutra
- Usa tentativa e erro quando a dedução lógica não for suficiente

Exemplos:
- <move>C7=3</move> coloca 3 na linha C, coluna 7
- <move>A1=0</move> remove o número de A1 (retroceder)"""

THINK_SUDOKU_SYSTEM_PROMPT = """Estás a resolver um puzzle de Sudoku passo a passo com raciocínio cuidadoso.

REGRAS:
- Preenche cada célula vazia (.) com um número de 1-9
- Cada linha deve conter todos os números 1-9 exatamente uma vez
- Cada coluna deve conter todos os números 1-9 exatamente uma vez  
- Cada caixa 3x3 deve conter todos os números 1-9 exatamente uma vez
- Faz UMA jogada de cada vez

Pensa passo a passo dentro das tags <think>...</think> sobre:
1. Quais células vazias têm menos possibilidades
2. Que números faltam em cada linha, coluna e caixa
3. Quais células podem conter apenas um número específico
4. Se precisas de retroceder de jogadas anteriores
5. O teu raciocínio lógico para a jogada

FORMATO DA JOGADA:
DEVES sempre terminar com exatamente uma jogada neste formato: <move>A1=5</move>
- A-I representa as linhas (A=linha superior, I=linha inferior)
- 1-9 representa as colunas (1=mais à esquerda, 9=mais à direita)
- O número depois de = é o que queres colocar
- Se ficares bloqueado, podes retroceder definindo uma célula para 0: <move>A1=0</move>

ESTRATÉGIA:
- Usa dedução lógica primeiro (singles isolados, singles escondidos)
- Se não houver jogadas lógicas disponíveis, considera retroceder
- Depois de retroceder, tenta números diferentes
- Usa tentativa e erro quando necessário

Exemplo:
<think>
A observar a linha A, coluna 1. A linha A está a faltar 2, 4, 8.
A coluna 1 já tem 2 e 8, então apenas 4 é possível.
A caixa 1 ainda não tem 4, portanto A1 deve ser 4.
</think>

<move>A1=4</move>"""