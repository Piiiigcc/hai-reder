"""Render template fields once and keep Jira code blocks inside their endpoint."""

import re


_FIELD = re.compile(
    r"(?P<fence>`{3,})text\n\{\{(?P<endpoint>issue_[ij]_text)\}\}\n(?P=fence)"
    r"|\{\{(?P<name>\w+)\}\}"
)


def render_role_template(template, row, **fields):
    fields.update(
        issue_i_text=row["issue_i"].get("text", ""),
        issue_j_text=row["issue_j"].get("text", ""),
    )

    def substitute(match):
        if match["endpoint"] is None:
            return fields.get(match["name"], match[0])
        value = fields[match["endpoint"]]
        longest = max((len(s) for s in re.findall(r"`+", value)), default=0)
        fence = "`" * max(len(match["fence"]), longest + 1)
        return f"{fence}text\n{value}\n{fence}"

    return _FIELD.sub(substitute, template).rstrip() + "\n\nJSON Output:\n"
