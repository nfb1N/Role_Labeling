# Role Labeling

Role Labeling is a small Python project for extracting process-relevant action units from textual process descriptions. It uses Stanza to normalize and parse English text, then applies dependency-based rules to identify predicates such as root, coordinated, and subordinate actions.

## Project Structure

- `main.py` - demo entry point that runs preprocessing and action-unit extraction on sample text.
- `preprocessing.py` - text normalization and Stanza-based linguistic preprocessing.
- `action_units.py` - rule-based action-unit extraction from parsed sentences.
- `models.py` - dataclasses used by the preprocessing and extraction pipeline.

## Requirements

- Python 3.10 or newer
- `stanza`
- Stanza English model data

Install the Python dependency:

```bash
pip install stanza
```

Download the English Stanza model before the first run:

```bash
python -c "import stanza; stanza.download('en')"
```

## Usage

Run the demo:

```bash
python main.py
```

The script prints JSON action-unit records. Each record includes the sentence id, predicate token, lemma, dependency relation, inferred action type, parent token information, and source sentence text.

## Pipeline

1. `preprocess_text()` normalizes raw text and builds a structured representation of sentences and tokens.
2. `extract_action_units()` identifies process-relevant predicates from the dependency parse.
3. `action_units_to_dict()` converts the result into JSON-friendly dictionaries.

## Example Output

```json
[
  {
    "action_unit_id": "S1_AU1",
    "sentence_id": 1,
    "predicate_text": "submits",
    "predicate_lemma": "submit",
    "predicate_type": "root_action"
  }
]
```

The actual output contains additional fields and depends on the Stanza parse for the input text.
