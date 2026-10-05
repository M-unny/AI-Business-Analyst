# AI Business Analytics & Decision Intelligence Platform

Work in progress. Milestone 1 (upload + data profiler) is implemented.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
python main.py datasets/sample_sales.csv
pytest
```
