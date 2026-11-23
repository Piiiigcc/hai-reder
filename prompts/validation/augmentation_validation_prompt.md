Independently evaluate the proposed Jira rewrites. Target relation is {{target_label}}.

Original pair
{{seed_pair_json}}

Generated pair
{{generated_pair_json}}

Check that both texts are plausible requirements, preserve the source constraints, and still exhibit the target relationship. Return exactly one JSON object with the boolean fields "is_valid", "label_matches", and a "rationale" of at most 15 words. Use false when evidence is insufficient.
