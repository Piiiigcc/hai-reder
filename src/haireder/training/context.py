"""Allocate a shared token budget to both requirements while preserving role instructions and arguments."""

import re

_SECTIONS = re.compile(
    r"^Requirement [AB]:\s*\n(?P<fence>`{3,})text\n(?P<body>.*?)\n(?P=fence)(?=\n|$)",
    re.M | re.S,
)


def fit_prompt(prompt, tokenizer, budget):
    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    if len(encode(prompt)) <= budget:
        return prompt
    parts = list(_SECTIONS.finditer(prompt))
    if len(parts) != 2:
        raise ValueError(
            f"Prompt exceeds its {budget}-token budget and has no unambiguous endpoint sections"
        )
    tokens = [encode(match["body"]) for match in parts]

    def render(cap):
        shortened = []
        for value, match in zip(tokens, parts):
            if len(value) <= cap:
                shortened.append(match["body"])
            else:
                head = (cap + 1) // 2
                tail = cap - head
                shortened.append(
                    tokenizer.decode(value[:head])
                    + " […] "
                    + tokenizer.decode(value[-tail:] if tail else [])
                )
        return (
            prompt[: parts[0].start("body")]
            + shortened[0]
            + prompt[parts[0].end("body") : parts[1].start("body")]
            + shortened[1]
            + prompt[parts[1].end("body") :]
        )

    lo, hi = 16, max(map(len, tokens))
    best = None
    while lo <= hi:
        cap = (lo + hi) // 2
        candidate = render(cap)
        if len(encode(candidate)) <= budget:
            best, lo = candidate, cap + 1
        else:
            hi = cap - 1
    if best is None:
        raise ValueError(f"Instructions and debate evidence exceed {budget} tokens")
    return best
