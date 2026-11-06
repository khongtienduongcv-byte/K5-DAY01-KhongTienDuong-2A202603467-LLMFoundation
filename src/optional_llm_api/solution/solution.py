"""
Day 1 — LLM API Foundation
AICB-P1: AI Practical Competency Program, Phase 1

Instructions:
    1. Fill in every section marked with TODO.
    2. Do NOT change function signatures.
    3. Copy this file to src/optional_llm_api/solution/solution.py when done.
    4. Run from the repo root: pytest tests/test_solution.py -v

Models (11/2026): gpt-4o / gpt-4o-mini are removed from the OpenAI API on 2027-01-14
(https://developers.openai.com/api/docs/deprecations), so this lab uses gpt-6.1-terra / gpt-6-luna.
The result keys and table headers below still say "gpt4o" / "GPT-4o" / "Mini": they are a test
contract kept for compatibility. Read "gpt4o" as "the large model" and "mini" as "the small model".
"""

import os
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Estimated costs per 1K OUTPUT tokens (USD) — OpenAI pricing page, 11/2026. Update if pricing changes.
# ---------------------------------------------------------------------------
COST_PER_1K_OUTPUT_TOKENS = {
    "gpt-6.1-terra": 0.005,   # $5.00 / 1M output tokens
    "gpt-6-luna": 0.0005,     # $0.50 / 1M output tokens
}

OPENAI_MODEL = "gpt-6.1-terra"
OPENAI_MINI_MODEL = "gpt-6-luna"


# ---------------------------------------------------------------------------
# Task 1 — Call the large model (OPENAI_MODEL)
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API and return the response text + latency.

    Args:
        prompt:      The user message to send.
        model:       The OpenAI model to use (default: OPENAI_MODEL).
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            reasoning_effort="none",          # GPT-6 models only accept temperature/top_p with effort "none"
            temperature=temperature, top_p=top_p,
            max_completion_tokens=max_tokens,  # GPT-6 models reject max_tokens
        )
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",           # GPT-6 models only accept temperature/top_p with effort "none"
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,  # GPT-6 models reject max_tokens
    )
    latency = time.perf_counter() - start

    response_text = response.choices[0].message.content
    return response_text, latency


# ---------------------------------------------------------------------------
# Task 2 — Call the small model (OPENAI_MINI_MODEL)
# ---------------------------------------------------------------------------
def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call the OpenAI Chat Completions API using OPENAI_MINI_MODEL and return the
    response text + latency.

    Args:
        prompt:      The user message to send.
        temperature: Sampling temperature (0.0 – 2.0).
        top_p:       Nucleus sampling threshold.
        max_tokens:  Maximum number of tokens to generate.

    Returns:
        A tuple of (response_text: str, latency_seconds: float).

    Hint:
        Reuse call_openai() by passing model=OPENAI_MINI_MODEL.
    """
    return call_openai(prompt, model=OPENAI_MINI_MODEL, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Task 3 — Compare the large vs the small model
# ---------------------------------------------------------------------------
def compare_models(prompt: str) -> dict:
    """
    Call both OPENAI_MODEL and OPENAI_MINI_MODEL with the same prompt and return a
    comparison dictionary.

    Args:
        prompt: The user message to send to both models.

    Returns:
        A dict with keys:
            - "gpt4o_response":      str
            - "mini_response":       str
            - "gpt4o_latency":       float
            - "mini_latency":        float
            - "gpt4o_cost_estimate": float  (estimated USD for the response)

    Hint:
        Cost estimate = (len(response.split()) / 0.75) / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]
        (0.75 words ≈ 1 token is a rough approximation)
    """
    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)

    def estimate_cost(response_text: str) -> float:
        tokens_estimate = len(response_text.split()) / 0.75
        return tokens_estimate / 1000 * COST_PER_1K_OUTPUT_TOKENS[OPENAI_MODEL]

    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": estimate_cost(gpt4o_response),
    }


# ---------------------------------------------------------------------------
# Task 4 — Streaming chatbot with conversation history
# ---------------------------------------------------------------------------
def streaming_chatbot() -> None:
    """
    Run an interactive streaming chatbot in the terminal.

    Behaviour:
        - Streams tokens from OpenAI as they arrive (print each chunk).
        - Maintains the last 3 conversation turns in history.
        - Typing 'quit' or 'exit' ends the loop.

    Hints:
        - Keep a list `history` of {"role": ..., "content": ...} dicts.
        - Use stream=True in client.chat.completions.create() and iterate:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                print(delta, end="", flush=True)
        - After each turn, append the assistant reply to history.
        - Trim history to the last 3 turns (= 6 messages): history = history[-6:]
    """
    # TODO: enter while-loop, read user input, stream response, maintain history
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    history: list[dict] = []

    print("=== Streaming chatbot (type 'quit' or 'exit') ===")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            break

        text = user_input.strip()
        if not text:
            continue
        if text.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        # Keep only the last 3 turns (= 6 messages).
        history.append({"role": "user", "content": text})
        history = history[-6:]

        print("Assistant: ", end="", flush=True)
        stream = client.chat.completions.create(
            model=OPENAI_MINI_MODEL,
            messages=history,
            stream=True,
        )
        full_response = ""
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            print(delta, end="", flush=True)
            full_response += delta
        print()

        # After each turn, append the assistant reply to history.
        history.append({"role": "assistant", "content": full_response})
        history = history[-6:]


# ---------------------------------------------------------------------------
# Bonus Task A — Retry with exponential backoff
# ---------------------------------------------------------------------------
def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """
    Call fn(). If it raises an exception, retry up to max_retries times
    with exponential backoff (base_delay * 2^attempt).

    Args:
        fn:          Zero-argument callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay:  Initial delay in seconds before the first retry.

    Returns:
        The return value of fn() on success.

    Raises:
        The last exception raised by fn() after all retries are exhausted.
    """
    # TODO: implement retry loop with exponential backoff
    attempt = 0
    while True:
        try:
            return fn()
        except Exception:
            if attempt >= max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# Bonus Task B — Batch compare
# ---------------------------------------------------------------------------
def batch_compare(prompts: list[str]) -> list[dict]:
    """
    Run compare_models on each prompt in the list.

    Args:
        prompts: List of prompt strings.

    Returns:
        List of dicts, each being the compare_models result with an extra
        key "prompt" containing the original prompt string.
    """
    # TODO: iterate over prompts, call compare_models, add "prompt" key
    results: list[dict] = []
    for prompt in prompts:
        result = dict(compare_models(prompt))
        result["prompt"] = prompt
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# Bonus Task C — Format comparison table
# ---------------------------------------------------------------------------
def format_comparison_table(results: list[dict]) -> str:
    """
    Format a list of compare_models results as a readable text table.

    Args:
        results: List of dicts as returned by batch_compare.

    Returns:
        A formatted string table with columns:
        Prompt | GPT-4o Response | Mini Response | GPT-4o Latency | Mini Latency

    Hint:
        Truncate long text to 40 characters for readability.
    """
    # TODO: build and return a formatted table string
    header = f"{'Prompt':<40} | {'GPT-4o Response':<40} | {'Mini Response':<40} | {'GPT-4o Latency':>15} | {'Mini Latency':>13}"
    separator = "-" * len(header)
    lines = [header, separator]

    def truncate(text: str, limit: int = 40) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= limit else text[: limit - 3] + "..."

    for result in results:
        lines.append(
            f"{truncate(result.get('prompt', '')):<40} | "
            f"{truncate(result.get('gpt4o_response', '')):<40} | "
            f"{truncate(result.get('mini_response', '')):<40} | "
            f"{result.get('gpt4o_latency', 0.0):>15.3f} | "
            f"{result.get('mini_latency', 0.0):>13.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point for manual testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_prompt = "Explain the difference between temperature and top_p in one sentence."
    print("=== Comparing models ===")
    result = compare_models(test_prompt)
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Starting chatbot (type 'quit' to exit) ===")
    streaming_chatbot()
