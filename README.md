# Example usage

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run script without evaluation
python3 llm_parse.py --eval False --input "data.txt" --output "parsing.txt"
```

## Hypotheses

Hypothesis 0: Despite AI hallucinations, properly prompted LLMs can be used for syntax parsing without additional functionality (to prove this, we want to collect data on their mistakes)

Hypothesis 1: Due to numerous hallucinations, LLMs can be used for syntax parsing only as natural language interfaces, providing enhanced visualization, explanation and advanced data representation 

### Hypothesis 0

Script for LLM prompting
Evaluate script

- Groq -> Gemma

TODO:
1. run locally
2. evaluation script

### Hypothesis 1 

LLM as a natural language interface

- Functions for parsing + Script from Hypothesis

TODO:
1. def parsing() -> nltk, spacy
2. def generate_grammar() -> nltk
3. def visualize(): LLM generates code for visualization
