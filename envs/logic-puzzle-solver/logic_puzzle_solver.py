import json
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
from itertools import permutations, combinations
from collections import defaultdict

import verifiers as vf
from datasets import Dataset
from verifiers.types import Messages, State

AMALIA_SYSTEM = "O teu nome é Amália, e és um modelo avançado de linguagem útil. Responde sempre na língua do utilizador, a menos que sejas instruído em contrário, e lembra-te que a tua língua principal é o português europeu."


@dataclass
class LogicPuzzle:
    """Represents a logic puzzle with entities, attributes, and clues"""
    entities: List[str]  # e.g., ["Alice", "Bob", "Charlie"]
    attributes: Dict[str, List[str]]  # e.g., {"color": ["red", "blue", "green"], "pet": ["cat", "dog", "bird"]}
    clues: List[str]
    solution: Dict[str, Dict[str, str]]  # e.g., {"Alice": {"color": "red", "pet": "cat"}}
    difficulty: str  # "easy", "medium", "hard"


class PuzzleGenerator:
    """Generates random logic puzzles of varying difficulty"""
    
    NAMES = ["Ana", "Bruno", "Carlos", "Diana", "Eduardo", "Filipa", "Gonçalo", "Helena"] 
    COLORS = ["vermelho", "azul", "verde", "amarelo", "roxo", "laranja", "rosa", "castanho"] 
    PETS = ["gato", "cão", "pássaro", "peixe", "coelho", "hamster", "tartaruga", "cobra"]
    DRINKS = ["café", "chá", "sumo", "água", "leite", "refrigerante", "batido", "limonada"]
    HOBBIES = ["leitura", "jogar videojogos", "cozinhar", "jardinagem", "pintura", "dança", "caminhada", "natação"]
    FRUITS = ["maçã", "banana", "laranja", "uva", "morango", "manga", "pêssego", "cereja"]
    
    # Translation mappings for attribute types
    ATTR_TRANSLATIONS = {
        "color": "cor",
        "pet": "animal",
        "drink": "bebida",
        "hobby": "hobby",
        "fruit": "fruta"
    }
    
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
    
    def generate_puzzle(self, difficulty: str = "medium") -> LogicPuzzle:
        """Generate a random logic puzzle"""
        if difficulty == "easy":
            num_entities = 3
            num_attributes = 2
            num_clues = 5
        elif difficulty == "medium":
            num_entities = 4
            num_attributes = 3
            num_clues = 8
        else:  # hard
            num_entities = 5
            num_attributes = 4
            num_clues = 12
        
        # Select random entities and attributes
        entities = random.sample(self.NAMES, num_entities)
        
        attribute_pools = {
            "cor": self.COLORS,
            "animal": self.PETS,
            "bebida": self.DRINKS,
            "hobby": self.HOBBIES,
            "fruta": self.FRUITS
        }
        
        selected_attr_types = random.sample(list(attribute_pools.keys()), num_attributes)
        attributes = {}
        for attr_type in selected_attr_types:
            attributes[attr_type] = random.sample(attribute_pools[attr_type], num_entities)
        
        # Generate a random valid solution
        solution = {}
        for i, entity in enumerate(entities):
            solution[entity] = {}
            for attr_type, attr_values in attributes.items():
                # Ensure each attribute value is used exactly once
                available = [v for v in attr_values if not any(
                    v == sol_data.get(attr_type) for sol_data in solution.values()
                )]
                if available:
                    solution[entity][attr_type] = random.choice(available)
                else:
                    # This shouldn't happen with proper generation
                    solution[entity][attr_type] = attr_values[i]
        
        # Generate clues based on the solution
        clues = self._generate_clues(entities, attributes, solution, num_clues)
        
        return LogicPuzzle(
            entities=entities,
            attributes=attributes,
            clues=clues,
            solution=solution,
            difficulty=difficulty
        )
    
    def _get_attr_plural(self, attr_type: str) -> str:
        """Get plural form of attribute type in Portuguese"""
        plurals = {
            "cor": "cores",
            "animal": "animais",
            "bebida": "bebidas",
            "hobby": "hobbies",
            "fruta": "frutas"
        }
        return plurals.get(attr_type, attr_type + "s")
    
    def _generate_clues(self, entities: List[str], attributes: Dict[str, List[str]], 
                        solution: Dict[str, Dict[str, str]], num_clues: int) -> List[str]:
        """Generate clues that uniquely determine the solution"""
        clues = []
        clue_types = [
            self._direct_clue,
            self._negative_clue,
            self._relationship_clue,
            self._either_or_clue,
            self._combination_clue
        ]
        
        attempts = 0
        while len(clues) < num_clues and attempts < num_clues * 3:
            clue_func = random.choice(clue_types)
            clue = clue_func(entities, attributes, solution)
            if clue and clue not in clues:
                clues.append(clue)
            attempts += 1
        
        # Ensure we have enough clues
        while len(clues) < num_clues:
            # Add more direct clues if needed
            entity = random.choice(entities)
            attr_type = random.choice(list(attributes.keys()))
            value = solution[entity][attr_type]
            
            # Use appropriate article based on attribute type and value
            article = self._get_article(attr_type, value)
            clue = f"{entity} tem {article} {attr_type} {value}."
            if clue not in clues:
                clues.append(clue)
        
        random.shuffle(clues)
        return clues
    
    def _get_article(self, attr_type: str, value: str) -> str:
        """Get appropriate article (o/a) based on gender of attribute"""
        # Feminine attributes
        if attr_type in ["cor", "bebida", "fruta"]:
            return "a"
        # Masculine attributes
        elif attr_type in ["animal", "hobby"]:
            return "o"
        return "o"
    
    def _direct_clue(self, entities, attributes, solution):
        """Generate a direct clue: 'Alice tem a cor vermelha.'"""
        entity = random.choice(entities)
        attr_type = random.choice(list(attributes.keys()))
        value = solution[entity][attr_type]
        article = self._get_article(attr_type, value)
        return f"{entity} tem {article} {attr_type} {value}."
    
    def _negative_clue(self, entities, attributes, solution):
        """Generate a negative clue: 'Bob não tem o gato.'"""
        entity = random.choice(entities)
        attr_type = random.choice(list(attributes.keys()))
        correct_value = solution[entity][attr_type]
        wrong_values = [v for v in attributes[attr_type] if v != correct_value]
        if wrong_values:
            wrong_value = random.choice(wrong_values)
            article = self._get_article(attr_type, wrong_value)
            return f"{entity} não tem {article} {attr_type} {wrong_value}."
        return None
    
    def _relationship_clue(self, entities, attributes, solution):
        """Generate a relationship clue: 'A pessoa com o gato tem a cor azul.'"""
        if len(list(attributes.keys())) < 2:
            return None
        attr_type1, attr_type2 = random.sample(list(attributes.keys()), 2)
        entity = random.choice(entities)
        value1 = solution[entity][attr_type1]
        value2 = solution[entity][attr_type2]
        article1 = self._get_article(attr_type1, value1)
        article2 = self._get_article(attr_type2, value2)
        return f"A pessoa com {article1} {attr_type1} {value1} tem {article2} {attr_type2} {value2}."
    
    def _either_or_clue(self, entities, attributes, solution):
        """Generate an either/or clue: 'Ou Alice ou Bob tem o cão.'"""
        if len(entities) < 2:
            return None
        entity1, entity2 = random.sample(entities, 2)
        attr_type = random.choice(list(attributes.keys()))
        value1 = solution[entity1][attr_type]
        value2 = solution[entity2][attr_type]
        
        # Pick one of them who actually has something
        if random.choice([True, False]):
            article = self._get_article(attr_type, value1)
            return f"Ou {entity1} ou {entity2} tem {article} {attr_type} {value1} (mas não ambos)."
        else:
            article = self._get_article(attr_type, value2)
            return f"Ou {entity1} ou {entity2} tem {article} {attr_type} {value2} (mas não ambos)."
    
    def _combination_clue(self, entities, attributes, solution):
        """Generate a combination clue: 'Alice e Bob juntos têm o gato e o cão.'"""
        if len(entities) < 2 or not attributes:
            return None
        entity1, entity2 = random.sample(entities, 2)
        attr_type = random.choice(list(attributes.keys()))
        value1 = solution[entity1][attr_type]
        value2 = solution[entity2][attr_type]
        article1 = self._get_article(attr_type, value1)
        article2 = self._get_article(attr_type, value2)
        attr_plural = self._get_attr_plural(attr_type)
        return f"{entity1} e {entity2} juntos têm {article1}s {attr_plural} {value1} e {value2}."


def load_environment(
    num_puzzles: int = 10000,
    difficulty_distribution: Dict[str, float] = None,
    max_turns: int = 10,
    min_turns: int = 3,
    seed: int = 42,
    **env_args,
) -> vf.Environment:
    """
    Load the logic puzzle solver environment.
    
    Args:
        num_puzzles: Number of puzzles to generate
        difficulty_distribution: Dict mapping difficulty to probability, e.g. {"easy": 0.3, "medium": 0.5, "hard": 0.2}
        max_turns: Maximum number of turns allowed
        min_turns: Minimum number of turns before solution attempt is allowed
        seed: Random seed for reproducibility
    """
    
    if difficulty_distribution is None:
        difficulty_distribution = {"easy": 0.3, "medium": 0.4, "hard": 0.3}
    
    # Normalize distribution
    total = sum(difficulty_distribution.values())
    difficulty_distribution = {k: v/total for k, v in difficulty_distribution.items()}
    
    generator = PuzzleGenerator(seed=seed)
    
    def build_dataset() -> Dataset:
        """Build dataset of logic puzzles"""
        data = []
        
        for i in range(num_puzzles):
            # Choose difficulty based on distribution
            difficulty = random.choices(
                list(difficulty_distribution.keys()),
                weights=list(difficulty_distribution.values())
            )[0]
            
            puzzle = generator.generate_puzzle(difficulty)
            
            # Build attribute list text
            attr_lines = []
            for attr_type, values in puzzle.attributes.items():
                attr_plural = generator._get_attr_plural(attr_type)
                attr_lines.append(f"{attr_plural.capitalize()}: {', '.join(values)}")
            
            # Create the initial prompt in Portuguese
            initial_prompt = f"""Estás a resolver um puzzle de lógica. Aqui estão as entidades e atributos:

Entidades: {', '.join(puzzle.entities)}
{chr(10).join(attr_lines)}

Cada entidade tem exatamente um valor para cada tipo de atributo, e cada valor é atribuído a exatamente uma entidade.

Aqui estão as pistas:
{chr(10).join([f"{i+1}. {clue}" for i, clue in enumerate(puzzle.clues)])}

Podes:
1. Fazer deduções e acompanhar o teu raciocínio
2. Fazer perguntas de esclarecimento sobre as pistas
3. Pedir dicas se estiveres bloqueado
4. Submeter a tua resposta final quando estiveres pronto

Usa este formato para as respostas:
<think>
O teu raciocínio dedutivo passo a passo
</think>

<deducoes>
Lista de factos que determinaste (ex: "Alice tem a cor vermelha")
</deducoes>

<pergunta>
Qualquer pergunta de esclarecimento (opcional)
</pergunta>

<resposta>
Solução final (apenas quando estiveres confiante):
- Entidade1: atributo1=valor1, atributo2=valor2, ...
- Entidade2: atributo1=valor1, atributo2=valor2, ...
</resposta>"""
            
            data.append({
                "prompt": [{"role": "system", "content": AMALIA_SYSTEM},{"role": "user", "content": initial_prompt}],
                "answer": json.dumps(puzzle.solution),
                "task": "logic_puzzle",
                "info": {
                    "logic_puzzle": puzzle.__dict__,
                    "difficulty": difficulty,
                    "max_turns": max_turns,
                    "min_turns": min_turns,
                    "turns_taken": 0,
                    "deductions_made": [],
                    "contradictions": 0,
                    "hints_used": 0
                }
            })
        
        return Dataset.from_list(data)
    
    class LogicPuzzleEnv(vf.MultiTurnEnv):
        """Multi-turn environment for logic puzzle solving"""
        
        def is_completed(self, messages: Messages, state: State, **kwargs) -> bool:
            """Check if puzzle is solved or max turns reached"""
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            
            # Check if max turns reached
            if len(assistant_messages) >= state["info"]["max_turns"]:
                return True
            
            # Check if final answer was submitted
            if assistant_messages:
                last_message = assistant_messages[-1]["content"]
                if "<resposta>" in last_message and "</resposta>" in last_message:
                    return True
            
            return False
        
        def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
            """Generate environment response based on model's action"""
            assistant_messages = [m for m in messages if m["role"] == "assistant"]
            state["info"]["turns_taken"] = len(assistant_messages)
            
            if not assistant_messages:
                return [], state
            
            last_message = assistant_messages[-1]["content"]
            response = ""
            
            # Parse the last message
            if "<pergunta>" in last_message and "</pergunta>" in last_message:
                # Answer clarifying questions
                question_start = last_message.find("<pergunta>") + 10
                question_end = last_message.find("</pergunta>")
                question = last_message[question_start:question_end].strip()
                
                # Provide clarification (simplified - could be more sophisticated)
                response = f"Esclarecimento: Cada pista indica um facto sobre o puzzle. Todas as pistas são verdadeiras e consistentes. {question} - Sim, esta interpretação está correta se não contradizer outras pistas."
            
            elif "<resposta>" in last_message and "</resposta>" in last_message:
                # Final answer submitted - provide feedback
                response = "A tua resposta foi submetida. O puzzle está agora completo."
            
            else:
                # Track deductions
                if "<deducoes>" in last_message and "</deducoes>" in last_message:
                    ded_start = last_message.find("<deducoes>") + 10
                    ded_end = last_message.find("</deducoes>")
                    deductions = last_message[ded_start:ded_end].strip()
                    state["info"]["deductions_made"].append(deductions)
                
                # Provide encouragement or hints based on progress
                turns_left = state["info"]["max_turns"] - state["info"]["turns_taken"]
                if turns_left <= 2:
                    response = f"Tens {turns_left} jogadas restantes. Considera submeter a tua resposta final se estiveres confiante."
                elif state["info"]["turns_taken"] >= state["info"]["min_turns"] and random.random() < 0.3:
                    response = "Estás a fazer bom progresso. Continua com as tuas deduções ou submete a tua resposta se estiveres pronto."
                else:
                    response = "Continua com o teu raciocínio. O que podes deduzir das pistas?"
            
            return [{"role": "user", "content": response}], state
    
    # Create scoring rubric
    def correctness_score(completion: List[dict], state: dict, **kwargs) -> float:
        """Score based on correctness of final answer"""
        puzzle_solution = json.loads(state["answer"])
        
        # Find the final answer in the completion
        for message in reversed(completion):
            if message["role"] == "assistant" and "<resposta>" in message["content"]:
                answer_start = message["content"].find("<resposta>") + 10
                answer_end = message["content"].find("</resposta>")
                if answer_end > answer_start:
                    answer_text = message["content"][answer_start:answer_end].strip()
                    
                    # Parse the submitted answer
                    submitted = {}
                    for line in answer_text.split("\n"):
                        if ":" in line and "=" in line:
                            parts = line.split(":")
                            if len(parts) == 2:
                                entity = parts[0].strip().strip("-").strip()
                                attrs = parts[1].strip()
                                submitted[entity] = {}
                                for attr_pair in attrs.split(","):
                                    if "=" in attr_pair:
                                        attr_parts = attr_pair.strip().split("=")
                                        if len(attr_parts) == 2:
                                            attr_type = attr_parts[0].strip()
                                            attr_value = attr_parts[1].strip()
                                            submitted[entity][attr_type] = attr_value
                    
                    # Calculate accuracy
                    if not submitted:
                        return 0.0
                    
                    correct_count = 0
                    total_count = 0
                    
                    for entity, attrs in puzzle_solution.items():
                        for attr_type, correct_value in attrs.items():
                            total_count += 1
                            if entity in submitted and attr_type in submitted[entity]:
                                if submitted[entity][attr_type] == correct_value:
                                    correct_count += 1
                    
                    return correct_count / total_count if total_count > 0 else 0.0
        
        return 0.0  # No answer submitted
    
    def efficiency_score(completion: List[dict], state: dict, **kwargs) -> float:
        """Score based on efficiency (fewer turns is better)"""
        turns_taken = state["info"]["turns_taken"]
        max_turns = state["info"]["max_turns"]
        min_turns = state["info"]["min_turns"]
        
        if turns_taken <= min_turns:
            return 1.0  # Perfect efficiency
        elif turns_taken >= max_turns:
            return 0.0  # Took maximum turns
        else:
            # Linear decay between min and max
            return 1.0 - (turns_taken - min_turns) / (max_turns - min_turns)
    
    def reasoning_quality_score(completion: List[dict], state: dict, **kwargs) -> float:
        """Score based on quality of reasoning (deductions, no contradictions)"""
        score = 0.0
        
        # Check for reasoning tags (Portuguese version)
        has_reasoning = any("<think>" in m["content"] for m in completion if m["role"] == "assistant")
        has_deductions = any("<deducoes>" in m["content"] for m in completion if m["role"] == "assistant")
        
        if has_reasoning:
            score += 0.5
        if has_deductions:
            score += 0.5
        
        # Bonus for systematic approach (checking multiple deductions)
        num_deduction_blocks = sum(1 for m in completion if m["role"] == "assistant" and "<deducoes>" in m["content"])
        if num_deduction_blocks >= 2:
            score += 0.5
        
        # Normalize to [0, 1]
        return min(score / 1.5, 1.0)
    
    def format_compliance_score(completion: List[dict], state: dict, **kwargs) -> float:
        """Score based on following the required format"""
        parser = vf.XMLParser(["think", "deducoes", "pergunta", "resposta"])
        
        total_messages = sum(1 for m in completion if m["role"] == "assistant")
        compliant_messages = 0
        
        for message in completion:
            if message["role"] == "assistant":
                # Check if message uses at least one of the expected tags (Portuguese version)
                content = message["content"]
                if any(f"<{tag}>" in content for tag in ["think", "deducoes", "pergunta", "resposta"]):
                    compliant_messages += 1
        
        return compliant_messages / total_messages if total_messages > 0 else 0.0
    
    # Create rubric with weighted scoring
    rubric = vf.Rubric(
        funcs=[
            correctness_score,
            efficiency_score,
            reasoning_quality_score,
            format_compliance_score
        ],
        weights=[0.5, 0.2, 0.2, 0.1]  # Correctness is most important
    )
    
    # Build dataset and create environment
    dataset = build_dataset()
    env = LogicPuzzleEnv(
        dataset=dataset,
        rubric=rubric,
        max_turns=max_turns
    )
    
    return env