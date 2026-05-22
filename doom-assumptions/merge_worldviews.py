#!/usr/bin/env python3
"""Merge worldviews-generated.json into data.js in place.

For each slug present in the generated file, locates the existing block
inside `worldviews: { ... }` in data.js and rewrites its description /
probabilities / ranges / reasoning sub-blocks. The `name`, `author`, and
`perspective` lines are left alone so editorial choices (e.g. "X-like"
naming) survive a regen.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data.js"
GEN = HERE / "worldviews-generated.json"


def find_block(text: str, slug: str) -> tuple[int, int] | None:
    """Return (start, end) of `<slug>: { ... }` in data.js, or None."""
    # The worldview block opens with `<slug>: {` at the start of some line
    # inside the `worldviews:` object. Find that, then balance braces.
    m = re.search(rf"^\s+{re.escape(slug)}: \{{", text, re.MULTILINE)
    if not m:
        return None
    start = m.start()
    # Walk braces from the opening `{`.
    brace_start = text.index("{", m.start())
    depth = 1
    i = brace_start + 1
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    # Include trailing comma if present (so the replacement looks clean).
    end = i
    if end < len(text) and text[end] == ",":
        end += 1
    return start, end


def format_prob(p: float) -> str:
    """Match the formatting used in the existing data.js (trim trailing zeros)."""
    if p == 0 or p == 1:
        return str(int(p))
    # Round to 4 decimals, strip trailing zeros.
    s = f"{p:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def render_block(slug: str, gen: dict, existing: str) -> str:
    """Build a replacement block string for `slug` using the generated data
    and preserving identity fields from the existing block.
    """
    # Pull preserved fields off the existing block (name, author, perspective).
    def grab(field: str) -> str:
        m = re.search(rf'{field}: ("[^"]*"),', existing)
        return m.group(1) if m else f'"{slug}-like"'

    name = grab("name")
    author = grab("author")
    perspective = grab("perspective")

    desc = gen.get("description", "").replace('"', '\\"')
    probs = gen.get("probabilities", {})
    ranges = gen.get("ranges", {})
    reasoning = gen.get("reasoning", {})

    keys = [
        "t-ai-inc",
        "t-d-no-inc",
        "t-single",
        "t-d-multi",
        "t-has-rep",
        "t-d-no-rep",
        "t-expects",
        "t-d-expects",
        "t-d-no-expects",
    ]

    prob_lines = ",\n".join(
        f'          "{k}": {format_prob(float(probs.get(k, 0.5)))}' for k in keys
    )
    range_lines = ",\n".join(
        f'          "{k}": {{ lo: {format_prob(float(ranges.get(k, {}).get("lo", 0)))}, hi: {format_prob(float(ranges.get(k, {}).get("hi", 1)))} }}'
        for k in keys
    )
    reason_lines = ",\n".join(
        f'          "{k}": {json.dumps(reasoning.get(k, ""), ensure_ascii=False)}'
        for k in keys
    )

    return f"""      {slug}: {{
        name: {name},
        author: {author},
        perspective: {perspective},
        description: "{desc}",
        probabilities: {{
{prob_lines}
        }},
        ranges: {{
{range_lines}
        }},
        reasoning: {{
{reason_lines}
        }}
      }},"""


def main():
    if not GEN.exists():
        print(f"No {GEN.name} found.", file=sys.stderr)
        sys.exit(1)

    generated = json.loads(GEN.read_text(encoding="utf-8"))
    text = DATA.read_text(encoding="utf-8")

    skipped_no_match = []
    skipped_error = []
    merged = []

    for slug, gen in generated.items():
        if "error" in gen:
            skipped_error.append(slug)
            continue
        loc = find_block(text, slug)
        if not loc:
            skipped_no_match.append(slug)
            continue
        start, end = loc
        existing_block = text[start:end]
        replacement = render_block(slug, gen, existing_block)
        text = text[:start] + replacement + text[end:]
        merged.append(slug)

    DATA.write_text(text, encoding="utf-8")
    print(f"merged: {', '.join(merged) if merged else '(none)'}")
    if skipped_no_match:
        print(f"skipped (no match in data.js): {', '.join(skipped_no_match)}")
    if skipped_error:
        print(f"skipped (generator error): {', '.join(skipped_error)}")


if __name__ == "__main__":
    main()
