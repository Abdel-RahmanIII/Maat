import json
from pathlib import Path

in_path = Path(r"d:\GUC\Semester 8\Maat\results\exp1\observer_strategist_tactician_A\results.jsonl")
out_path = Path(r"d:\GUC\Semester 8\Maat\results\exp1\observer_strategist_tactician_D\results.jsonl")
suffix = out_path.parent.name.rsplit("_", 1)[-1]


def rewrite_suffix(value: object) -> object:
    if not isinstance(value, str):
        return value
    parts = value.rsplit("_", 1)
    if len(parts) != 2:
        return value
    return f"{parts[0]}_{suffix}"

count = 0
with in_path.open('r', encoding='utf-8') as inf, out_path.open('w', encoding='utf-8') as outf:
    for line in inf:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        try:
            obj = json.loads(line_stripped)
        except Exception:
            # skip malformed lines
            continue
        # If any turn has is_valid == True, write the whole row
        turns = obj.get('turns', [])
        if any((t.get('is_valid') is True) for t in turns):
            obj['game_id'] = rewrite_suffix(obj.get('game_id'))
            obj['condition'] = suffix
            outf.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

print(f"Wrote {count} valid rows to {out_path}")
