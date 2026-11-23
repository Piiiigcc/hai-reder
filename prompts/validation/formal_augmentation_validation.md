Assess these two rewrites independently as an issue-link reviewer. The proposed relation is {{target_label}}, defined as {{target_label_definition}}.

Original issues
{{seed_pair_json}}

Rewritten issues
{{generated_pair_json}}

First compare each rewrite with its own source. A change in wording or omission of a stack trace is acceptable if the important entities, behaviour and constraints survive. Set "is_valid" to true only when both issues preserve their technical meaning without invented material.

Then determine whether the rewritten issues give evidence for the defined relation, including its direction where relevant. A relationship can be expressed by shared facts without naming the category. An available source label does not establish that a rewrite preserved it. Set "label_matches" to false if there is insufficient evidence or another relation fits better.

Return a JSON object with boolean "is_valid" and "label_matches", and a short "rationale" citing concrete evidence from the issues. Use JSON booleans, not strings.
