Rewrite both Jira issues while preserving the directed relation {{target_label}}.
Its definition is {{target_label_definition}}.

Identify the components, requested behaviour, conditions and technical facts that relate the two source issues. In each rewrite keep the details needed to tell the specific target relation from a looser general link. Preserve identifiers and meaningful literals when they carry evidence. You may reorganise the prose, shorten logs and remove formatting, but keep the substance of both requests. Use enough sentences for each issue. Do not add a new link, a new constraint, or evidence copied from a reference example.

Issue A
{{issue_i_text}}

Issue B
{{issue_j_text}}

Other pairs in this class, for comparison only
{{reference_pairs}}

Return one JSON object with exactly two nonempty string properties, "augmented_issue_i_text" and "augmented_issue_j_text". Start with the opening brace and end after the closing brace.
