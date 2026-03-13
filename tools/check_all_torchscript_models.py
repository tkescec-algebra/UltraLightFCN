import json
from pathlib import Path

import torch


BASE_DIR = Path("/work/tools/export_torchscript_10")
INDEX_PATH = BASE_DIR / "index.json"
DEFAULT_INPUT_SHAPE = (1, 3, 256, 256)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_ts_path(entry):
    if isinstance(entry, str):
        p = Path(entry)
        return p if p.is_absolute() else (BASE_DIR / p)

    if isinstance(entry, dict):
        for key in ("ts_path", "path", "output", "outfile", "torchscript", "file", "ts"):
            if key in entry and entry[key]:
                p = Path(entry[key])
                return p if p.is_absolute() else (BASE_DIR / p)

    return None


def summarize_output(y):
    if isinstance(y, torch.Tensor):
        return f"Tensor shape={tuple(y.shape)} device={y.device}"

    if isinstance(y, (list, tuple)):
        parts = []
        for i, item in enumerate(y):
            if isinstance(item, torch.Tensor):
                parts.append(f"{i}:{tuple(item.shape)}@{item.device}")
            else:
                parts.append(f"{i}:{type(item).__name__}")
        return f"{type(y).__name__}[{', '.join(parts)}]"

    if isinstance(y, dict):
        parts = []
        for k, item in y.items():
            if isinstance(item, torch.Tensor):
                parts.append(f"{k}:{tuple(item.shape)}@{item.device}")
            else:
                parts.append(f"{k}:{type(item).__name__}")
        return f"dict{{{', '.join(parts)}}}"

    return type(y).__name__


def move_output_to_cpu(y):
    if isinstance(y, torch.Tensor):
        return y.detach().cpu()
    if isinstance(y, list):
        return [move_output_to_cpu(v) for v in y]
    if isinstance(y, tuple):
        return tuple(move_output_to_cpu(v) for v in y)
    if isinstance(y, dict):
        return {k: move_output_to_cpu(v) for k, v in y.items()}
    return y


def main():
    print(f"Device: {DEVICE}")

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"index.json not found: {INDEX_PATH}")

    with INDEX_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        if "models" in data and isinstance(data["models"], list):
            entries = data["models"]
        elif "exports" in data and isinstance(data["exports"], list):
            entries = data["exports"]
        else:
            entries = []
            for v in data.values():
                if isinstance(v, list):
                    entries.extend(v)
    else:
        raise ValueError("Unsupported index.json format")

    print(f"[info] found {len(entries)} entries in {INDEX_PATH}")

    ok = 0
    failed = 0
    skipped = 0

    for i, entry in enumerate(entries, 1):
        ts_file = extract_ts_path(entry)
        name = entry.get("name", str(ts_file)) if isinstance(entry, dict) else str(ts_file)

        if ts_file is None:
            skipped += 1
            print(f"[{i:02d}] SKIP  no ts path found in entry: {entry}")
            continue

        if not ts_file.exists():
            failed += 1
            print(f"[{i:02d}] FAIL  missing file: {ts_file}")
            continue

        shape = DEFAULT_INPUT_SHAPE
        if isinstance(entry, dict):
            inp = entry.get("input", {})
            if isinstance(inp, dict) and "shape" in inp:
                s = inp["shape"]
                if isinstance(s, list) and len(s) == 4:
                    shape = tuple(s)

        try:
            model = torch.jit.load(str(ts_file), map_location=DEVICE)
            model.eval()
            model.to(DEVICE)

            x = torch.randn(*shape, device=DEVICE)

            with torch.no_grad():
                y = model(x)

            _ = move_output_to_cpu(y)

            ok += 1
            print(f"[{i:02d}] OK    {ts_file.name} -> {summarize_output(y)}")

        except Exception as e:
            failed += 1
            print(f"[{i:02d}] FAIL  {name}")
            print(f"      {type(e).__name__}: {e}")

    print()
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print()
    print("=== SUMMARY ===")
    print(f"OK      : {ok}")
    print(f"FAILED  : {failed}")
    print(f"SKIPPED : {skipped}")
    print(f"TOTAL   : {len(entries)}")


if __name__ == "__main__":
    main()