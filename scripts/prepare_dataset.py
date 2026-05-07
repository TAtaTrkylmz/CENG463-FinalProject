from pathlib import Path


def main() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/external").mkdir(parents=True, exist_ok=True)
    print("Dataset preparation placeholder. Project-specific preprocessing goes here.")


if __name__ == "__main__":
    main()
