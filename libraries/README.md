# libraries

Python helper libraries, model code, and utilities used in ChessAIThon.

Overview

- Contains Python modules and scripts for data processing, model training and inference. Example files at repository root (also present under `libraries/`): `chess_aux.py`, `chess_transforms.py`, `chessmarro_model.py`, `get_best_fuctions.py`.

Quick start (Python)

```bash
# create a virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r ../modelDeploy/requirements.txt
# or install project-specific requirements if provided
```

Run small tests

```bash
python3 test.py
python3 test_c.py
```

Notes

- Some scripts depend on native extensions or prebuilt models. Check `modelDeploy/` for the runtime `requirements.txt` and the `README.md` there for details about model deployment.
- If you plan to modify model training code, prefer creating isolated virtual environments to avoid dependency conflicts.
# ai-libraries