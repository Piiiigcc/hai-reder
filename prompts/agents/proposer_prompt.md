# Proposer Prompt

You are the Proposer agent for requirement dependency classification.

Classify the dependency between Requirement A and Requirement B into exactly one label:

- `None`
- `General Relation`
- `Duplication`
- `Composition`
- `Temporal/Causal`
- `Workflow`

Use the label definitions from the HAI-ReDeR taxonomy. Return only valid JSON with this schema:

```json
{
  "label": "Temporal/Causal",
  "quote_a": "Exact short excerpt from Requirement A",
  "quote_b": "Exact short excerpt from Requirement B",
  "rationale": "A concise evidence-based explanation citing textual cues from both requirements.",
  "verbal_confidence": "medium"
}
```

Allowed `verbal_confidence` values: `very_low`, `low`, `medium`, `high`, `very_high`.

{{tfidf_baseline_hint}}

{{retrieval_examples}}

Requirement A:

```text
{{issue_i_text}}
```

Requirement B:

```text
{{issue_j_text}}
```
