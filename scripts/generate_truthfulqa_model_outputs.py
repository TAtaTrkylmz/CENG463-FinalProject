import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_uncertainty.io import write_jsonl
from llm_uncertainty.prompts import memory_prompt
from llm_uncertainty.representations import NLIProbabilityExtractor, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TruthfulQA answers from a model and convert them into an evaluation set."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-output", default=None, help="Optional path for raw generations JSONL.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--provider", choices=["hf", "openai"], default="hf")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default=None, help="Fallback device for both generator and NLI.")
    parser.add_argument("--generation-device", default=None, help="Device for answer generation.")
    parser.add_argument("--nli-device", default=None, help="Device for NLI auto-labeling.")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--nli-model", default="FacebookAI/roberta-large-mnli")
    parser.add_argument("--nli-max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    return parser.parse_args()


def _load_truthfulqa(limit: int) -> list[dict]:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    dataset = dataset.select(range(min(limit, len(dataset))))
    return [dataset[index] for index in range(len(dataset))]


class HuggingFaceGenerator:
    def __init__(self, model_name: str, device: str | None, max_new_tokens: int, temperature: float) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self, question: str) -> str:
        prompt = memory_prompt(question)
        if getattr(self.tokenizer, "chat_template", None):
            # Instruct-tuned models (e.g. Qwen2.5-*-Instruct) are fine-tuned only on
            # chat-formatted conversations. Feeding them the raw completion-style
            # prompt below leaves them off-distribution and prone to regurgitating
            # instruction-tuning boilerplate ("You are an AI assistant...") instead
            # of answering. Routing through the tokenizer's own chat template keeps
            # the input in the format the model was actually trained on.
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        else:
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = self.tokenizer.decode(output_ids[0][encoded["input_ids"].shape[1] :], skip_special_tokens=True)
        return generated.strip()


class OpenAIGenerator:
    def __init__(self, model_name: str, max_new_tokens: int, temperature: float) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for provider=openai.")

    def generate(self, question: str) -> str:
        payload = json.dumps(
            {
                "model": self.model_name,
                "messages": [{"role": "user", "content": question}],
                "temperature": self.temperature,
                "max_completion_tokens": self.max_new_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API request failed: {exc.code} {detail}") from exc
        return body["choices"][0]["message"]["content"].strip()


def _knowledge(row: dict) -> str:
    references = [row["best_answer"]] + [value for value in row["correct_answers"] if value != row["best_answer"]]
    return "Reference answers: " + " | ".join(references)


def _auto_label(answer: str, row: dict, extractor: NLIProbabilityExtractor) -> tuple[int, dict[str, float]]:
    premise_correct = "Reference answers: " + " | ".join([row["best_answer"]] + row["correct_answers"])
    premise_incorrect = "Incorrect answers: " + " | ".join(row["incorrect_answers"])
    probs = extractor.predict([premise_correct, premise_incorrect], [answer, answer], batch_size=2)
    correct_entail = float(probs[0, 0])
    incorrect_entail = float(probs[1, 0])
    label = 0 if correct_entail >= incorrect_entail else 1
    return label, {
        "correct_entailment": correct_entail,
        "incorrect_entailment": incorrect_entail,
    }


def main() -> None:
    args = parse_args()
    rows = _load_truthfulqa(args.limit)
    generation_device = args.generation_device or args.device
    nli_device = args.nli_device or args.device
    if args.provider == "hf":
        generator = HuggingFaceGenerator(
            args.model_name,
            generation_device,
            args.max_new_tokens,
            args.temperature,
        )
    else:
        generator = OpenAIGenerator(args.model_name, args.max_new_tokens, args.temperature)

    extractor = NLIProbabilityExtractor(
        model_name=args.nli_model,
        device=nli_device,
        max_length=args.nli_max_length,
    )

    raw_rows = []
    eval_rows = []
    for index, row in enumerate(rows):
        answer = generator.generate(row["question"])
        label, diagnostics = _auto_label(answer, row, extractor)
        sample_prefix = f"truthfulqa-gen-{index:05d}"
        raw_rows.append(
            {
                "sample_id": sample_prefix,
                "question": row["question"],
                "generated_answer": answer,
                "generator_model": args.model_name,
                "provider": args.provider,
                **diagnostics,
            }
        )
        eval_rows.append(
            {
                "sample_id": sample_prefix,
                "original_sample_id": sample_prefix,
                "source_dataset": "truthfulqa_generated",
                "source_config": args.provider,
                "source_split": "validation",
                "question": row["question"],
                "knowledge": _knowledge(row),
                "candidate_answer": answer,
                "label": label,
                "label_name": "hallucinated" if label == 1 else "factual",
                "reference_answer": row["best_answer"],
                "generator_model": args.model_name,
                "topic": row["category"],
                "notes": f"provider={args.provider};type={row['type']}",
                **diagnostics,
            }
        )
        if (index + 1) % 20 == 0:
            print(f"Generated {index + 1}/{len(rows)} answers...")
            time.sleep(0.05)

    output_dir = Path(args.output_dir)
    write_jsonl(eval_rows, output_dir / f"{args.split}.jsonl")
    raw_output = Path(args.raw_output) if args.raw_output else output_dir / f"{args.split}_raw_generations.jsonl"
    write_jsonl(raw_rows, raw_output)
    print(f"Saved {len(eval_rows)} generated evaluation rows to {output_dir / f'{args.split}.jsonl'}")
    print(f"Saved raw generations to {raw_output}")


if __name__ == "__main__":
    main()
