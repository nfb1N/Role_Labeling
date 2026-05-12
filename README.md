# Role Labeling

Role Labeling is a Python project for a bachelor thesis algorithm:

> Determination of the domain objects roles in the relations in process descriptions.

The current implementation contains the proposed algorithm pipeline. The repository is structured so future baseline systems and evaluation code can be added separately from the proposed algorithm.

## Project Structure

```text
Role_Labeling/
├── main.py
├── src/
│   ├── proposed_algorithm/
│   │   ├── preprocessing.py
│   │   ├── action_units.py
│   │   ├── arguments.py
│   │   ├── semantic_roles.py
│   │   ├── thesis_roles.py
│   │   ├── final_output.py
│   │   ├── models.py
│   │   └── pipeline.py
│   ├── baselines/
│   ├── evaluation/
│   └── utils/
├── data/
├── outputs/
├── experiments/
│   └── run_proposed.py
└── tests/
```

## Pipeline

The proposed algorithm currently runs six steps:

1. `preprocessing.py` - normalizes raw text and runs Stanza tokenization, POS tagging, lemmatization, and dependency parsing.
2. `action_units.py` - extracts process-relevant action units from dependency predicates.
3. `arguments.py` - extracts actors, candidate objects, oblique arguments, nested arguments, and inherited actors.
4. `semantic_roles.py` - interprets extracted arguments as intermediate semantic roles using VerbNet plus conservative structural fallback.
5. `thesis_roles.py` - maps intermediate semantic roles to thesis-specific domain-object roles.
6. `final_output.py` - builds traceable final JSON output with summary, records, and unresolved records.

The public pipeline function is:

```python
from src.proposed_algorithm.pipeline import run_proposed_algorithm
```

## Requirements

Use Python 3.10 or newer.

Required Python packages:

```bash
pip install stanza nltk
```

Download the required model/data resources before the first run:

```bash
python3 -c "import stanza; stanza.download('en')"
python3 -c "import nltk; nltk.download('verbnet')"
```

## Usage

Run the default example through the root wrapper:

```bash
python3 main.py
```

Or run the proposed-algorithm experiment directly:

```bash
python3 experiments/run_proposed.py
```

Both commands print the final Step 6 JSON output.

## Output Shape

The final output contains:

- `summary` - counts by thesis-specific role.
- `records` - traceable domain-object role records.
- `unresolved_records` - records that could not be assigned a thesis role.

Example summary for the current sample text:

```json
{
  "Target / Affected object": 3,
  "Result / Output object": 1,
  "Source / Input object": 1,
  "Transferred object": 1,
  "Recipient-linked element": 1,
  "Support / Instrument object": 1
}
```

## Notes

- The proposed algorithm lives under `src/proposed_algorithm/`.
- `src/baselines/` and `src/evaluation/` are placeholders for future work.
- The implementation is intentionally traceable: final records include syntax, semantic role, thesis role, rule id, and a compact trace string.
