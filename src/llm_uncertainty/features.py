from __future__ import annotations

import math


def logprob_features(token_logprobs: list[float]) -> dict[str, float]:
    finite = [value for value in token_logprobs if math.isfinite(value)]
    if not finite:
        return {
            "token_count": 0.0,
            "mean_logprob": 0.0,
            "negative_mean_logprob": 0.0,
            "sum_logprob": 0.0,
            "min_logprob": 0.0,
            "max_logprob": 0.0,
            "perplexity": 0.0,
        }

    mean_logprob = sum(finite) / len(finite)
    negative_mean_logprob = -mean_logprob
    return {
        "token_count": float(len(finite)),
        "mean_logprob": float(mean_logprob),
        "negative_mean_logprob": float(negative_mean_logprob),
        "sum_logprob": float(sum(finite)),
        "min_logprob": float(min(finite)),
        "max_logprob": float(max(finite)),
        "perplexity": float(math.exp(min(negative_mean_logprob, 50.0))),
    }


def add_rag_features(memory_record: dict, context_record: dict) -> dict[str, float]:
    memory_nll = float(memory_record["negative_mean_logprob"])
    context_nll = float(context_record["negative_mean_logprob"])
    return {
        "memory_negative_mean_logprob": memory_nll,
        "context_negative_mean_logprob": context_nll,
        "context_improvement": memory_nll - context_nll,
        "absolute_context_delta": abs(memory_nll - context_nll),
    }

