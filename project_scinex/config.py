import os

# Output directory (relative to where main.py is run)
OUTPUT_DIR = "output"

# How many citing papers to pull from Semantic Scholar
CITATION_LIMIT = 10

# PDF_PATH is now supplied at runtime via CLI argument.
# This fallback is only used if you run main.py with no arguments.
PDF_PATH = os.environ.get("PDF_PATH", "")