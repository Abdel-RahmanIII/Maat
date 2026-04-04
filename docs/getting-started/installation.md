# Installation

## Prerequisites

- Python 3.11+
- pip
- Optional: Stockfish binary available in PATH or via `STOCKFISH_PATH`

## Create Environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dependency Sources

- `requirements.txt` is the fastest path for local setup.
- `pyproject.toml` defines project metadata and tool configuration.

## Optional Engine Setup

If Stockfish is not in PATH, define `STOCKFISH_PATH`:

```powershell
$env:STOCKFISH_PATH = "C:\path\to\stockfish.exe"
```

## Verify Setup

```powershell
python -m pytest -q
```

Expected baseline from current implementation: all tests pass, with one Stockfish integration test potentially skipped if engine is unavailable.
