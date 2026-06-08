from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

from llm_uncertainty.features import logprob_features
from llm_uncertainty.prompts import context_prompt, memory_prompt


@dataclass
class ScoredAnswer:
    tokens: list[str]
    token_logprobs: list[float]
    features: dict[str, float]


class LocalCausalLMScorer:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str | None = None,
        model_dtype: str = "auto",
        use_chat_template: bool = False,
    ) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_chat_template = use_chat_template
        dtype_map = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if model_dtype not in dtype_map:
            raise ValueError(f"Unsupported model_dtype={model_dtype!r}. Choose from {sorted(dtype_map)}.")
        model_kwargs = {}
        if dtype_map[model_dtype] is not None:
            dtype_argument = "dtype" if int(transformers_version.split(".", maxsplit=1)[0]) >= 5 else "torch_dtype"
            model_kwargs[dtype_argument] = dtype_map[model_dtype]
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def format_prompt(self, prompt: str) -> str:
        if not self.use_chat_template:
            return prompt
        if not self.tokenizer.chat_template:
            raise ValueError(
                f"{self.model_name} does not define a chat template; "
                "rerun without --use-chat-template."
            )
        if prompt.endswith("\nAnswer:"):
            prompt = prompt[: -len("\nAnswer:")]
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def score_candidate(self, prompt: str, candidate_answer: str) -> ScoredAnswer:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        answer_ids = self.tokenizer(
            " " + candidate_answer,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids

        input_ids = torch.cat([prompt_ids, answer_ids], dim=1).to(self.device)
        prompt_length = prompt_ids.shape[1]

        with torch.no_grad():
            logits = self.model(input_ids).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]
            target_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)

        answer_start = max(prompt_length - 1, 0)
        answer_logprobs = target_log_probs[0, answer_start:].detach().cpu().tolist()
        answer_token_ids = answer_ids[0].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(answer_token_ids)
        return ScoredAnswer(
            tokens=tokens,
            token_logprobs=[float(value) for value in answer_logprobs],
            features=logprob_features([float(value) for value in answer_logprobs]),
        )


def score_record(record: dict, scorer: LocalCausalLMScorer, mode: str) -> dict:
    if mode == "memory":
        prompt = memory_prompt(record["question"])
    elif mode == "context":
        prompt = context_prompt(record["knowledge"], record["question"])
    else:
        raise ValueError(f"Unsupported inference mode: {mode}")

    prompt = scorer.format_prompt(prompt)
    scored = scorer.score_candidate(prompt, record["candidate_answer"])
    return {
        **record,
        "inference_mode": mode,
        "model_name": scorer.model_name,
        "prompt": prompt,
        "generated_text": record["candidate_answer"],
        "tokens": scored.tokens,
        "token_logprobs": scored.token_logprobs,
        **scored.features,
    }
