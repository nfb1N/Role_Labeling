# Role Labeling

Role Labeling is a Python project for a bachelor thesis algorithm:

> Determination of the domain objects roles in the relations in process descriptions.

The current implementation contains the proposed algorithm pipeline. The repository is structured so future baseline systems and evaluation code can be added separately from the proposed algorithm.

## Proposed Algorithm

The proposed algorithm currently runs six steps:

1. `preprocessing.py` - normalizes raw text and runs Stanza tokenization, POS tagging, lemmatization, and dependency parsing.
2. `action_units.py` - extracts process-relevant action units from dependency predicates.
3. `arguments.py` - extracts actors, candidate objects, oblique arguments, nested arguments, and inherited actors.
4. `semantic_roles.py` - interprets extracted arguments as intermediate semantic roles using VerbNet plus conservative structural fallback.
5. `thesis_roles.py` - maps intermediate semantic roles to thesis-specific roles, including `Performer / Agent` for actors/agents.
6. `final_output.py` - builds traceable final JSON output with summary, records, and unresolved records.

The role assignment layer uses this thesis role inventory:

- `Performer / Agent`
- `Target / Affected object`
- `Result / Output object`
- `Source / Input object`
- `Transferred object`
- `Support / Instrument object`
- `Recipient-linked element`

The implementation is intentionally traceable: final records include syntax, semantic role, thesis role, rule id, and a compact trace string.

The public pipeline function is:

```python
from src.proposed_algorithm.pipeline import run_proposed_algorithm
```

## Requirements

Use Python 3.10 or newer.

Required Python packages:

```bash
pip install stanza nltk datasets
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

## Data Generation

Downloaded and generated datasets are ignored by Git. Recreate them locally when needed.

Download PETv11 raw data:

```bash
python3 src/evaluation/pet_loader.py
```

This downloads PETv11 from Hugging Face and writes:

- `data/raw/pet/petv11_relations.json`
- `data/raw/pet/petv11_token_classification.json`
- `data/raw/pet/petv11_inspection.json`

Build candidate role cases:

```bash
python3 src/evaluation/pet_candidate_builder.py
```

This reads `data/raw/pet/petv11_relations.json` and creates:

- `data/gold/candidate_role_cases_v1.csv`

Draft or generated evaluation CSV files under `data/gold/` are also ignored. Only manually maintained documentation, such as annotation guidelines, should be committed by default.

## Output Shape

The final output contains:

- `summary` - counts by thesis-specific role.
- `records` - traceable domain-object role records.
- `unresolved_records` - records that could not be assigned a thesis role.

Example summary for the current sample text:

```json
{
  "Performer / Agent": 5,
  "Target / Affected object": 3,
  "Result / Output object": 1,
  "Source / Input object": 1,
  "Transferred object": 1,
  "Recipient-linked element": 1,
  "Support / Instrument object": 1
}
```
