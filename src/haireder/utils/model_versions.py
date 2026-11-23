"""Immutable upstream revisions used by the published experiment and local controls."""
REVISIONS = {
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen/Qwen2.5-3B-Instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "sentence-transformers/all-MiniLM-L6-v2": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    "google-bert/bert-base-uncased": "86b5e0934494bd15c9632b12f734a8a67f723594",
    "mlx-community/Qwen2.5-7B-Instruct-4bit": "c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
    "mlx-community/Qwen2.5-3B-Instruct-4bit": "4f83f8f146fdf28b512a06562b671d7af4fab457",
    "HuggingFaceTB/SmolLM2-135M-Instruct": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
}


def revision_kwargs(name):
    return {"revision": REVISIONS[name]} if name in REVISIONS else {}
