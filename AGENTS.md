You are an expert Machine Learning and LLM engineer assisting on this repository. You proficient in Python and Pytorch frameworks. 

# Code quality

Contributing to this repository, follow the requirements and best practices:

0. When you implement change to a class, always run corresponding tests, including ones you just written yourself and old ones;
1. Ruff (line length 120), type hints, docstrings using Google convention for Python;
2. Try to keep functions under 50 lines on code, give descriptive, verb-first names;
3. Keep classes under 200 LoC: once your change requires more, move this to separate class or function;
4. Follow good practices and avoid antipattern where possible, including (but not limited to):
   a. no imports in function's body
   b. no def in def patterns


# Repo structure

This is monorepo. The root folder contains the PLAN.md file that describes what should be build. For each task in this file, we have dedicated folder with relevant source code.

# Stack

The repo uses:

- Pytorch 2.4 (already installed in the enviroment);
- `uv` for managing dependencies, with separate group for dev dependencies;
- `pytest` for testing
- `huggingface` is used to load model and pretrained weights in the first place.
- `python-dotenv` -- ALWAYS load .env in the root of the repo to obtain necessary env vars, including HF cache locations to avoid unnecessary downloads.
- `loguru` for logging;

# Running the code

We use `uv` to run the code. To run the code, go to dedicated service folder and run `uv run ...`. Example:

```bash
uv run pytest -v
```