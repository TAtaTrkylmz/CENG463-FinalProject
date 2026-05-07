from __future__ import annotations


def memory_prompt(question: str) -> str:
    return f"Answer the question.\n\nQuestion: {question}\nAnswer:"


def context_prompt(knowledge: str, question: str) -> str:
    return (
        "Use the provided knowledge to answer the question.\n\n"
        f"Knowledge: {knowledge}\n\n"
        f"Question: {question}\nAnswer:"
    )

