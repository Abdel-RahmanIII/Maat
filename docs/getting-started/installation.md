# Installation

## Prerequisites

- Python 3.11+
- pip
- A Google AI Studio API key (for LLM calls)
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

## Configure API Key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_google_ai_studio_api_key
STOCKFISH_PATH=C:\path\to\stockfish.exe
```

The `GOOGLE_API_KEY` is required for any condition that calls the LLM. The `STOCKFISH_PATH` is optional and only needed for full-game experiments against Stockfish.

## Verify Setup

```powershell
python -m pytest -q
```

Expected: 58 tests pass, with one Stockfish integration test potentially skipped if engine is unavailable.
