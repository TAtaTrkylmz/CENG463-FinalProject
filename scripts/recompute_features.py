import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_uncertainty.features import logprob_features

def recompute_file(filepath: Path):
    print(f"Recomputing {filepath}...")
    records = []
    with filepath.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            # Recompute logprob features
            if "token_logprobs" in record:
                feats = logprob_features(record["token_logprobs"])
                record.update(feats)
            records.append(record)
            
    with filepath.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Done recomputing {filepath}")

def main():
    base_dir = Path("results/scored")
    for file_path in base_dir.rglob("*.jsonl"):
        recompute_file(file_path)

if __name__ == "__main__":
    main()
