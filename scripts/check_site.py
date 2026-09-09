#!/usr/bin/env python3
"""Mechanical checks for the homepage.

Rule 1: no hyphen or dash of any kind in visible text, code samples included.
Rule 2: under 120 words of prose, code samples excluded.
Rule 3: the headline is exactly "Free verified AI inference."

Exit 0 when every rule holds, 1 otherwise, with each offence printed.
"""
from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

FORBIDDEN = {
    "-": "hyphen minus",
    "‐": "hyphen",
    "‑": "non breaking hyphen",
    "‒": "figure dash",
    "–": "en dash",
    "—": "em dash",
    "―": "horizontal bar",
    "−": "minus sign",
}
HEADLINE = "Free verified AI inference."
WORD_LIMIT = 120


class Visible(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip = 0
        self.code = 0
        self.prose: list[str] = []
        self.code_text: list[str] = []
        self.h1: list[str] = []
        self.in_h1 = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self.skip += 1
        if tag in ("pre", "code"):
            self.code += 1
        if tag == "h1":
            self.in_h1 = True

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self.skip -= 1
        if tag in ("pre", "code"):
            self.code -= 1
        if tag == "h1":
            self.in_h1 = False

    def handle_data(self, data: str) -> None:
        if self.skip:
            return
        if self.in_h1:
            self.h1.append(data)
        if self.code:
            self.code_text.append(data)
        else:
            self.prose.append(data)


def main(path: Path) -> int:
    parser = Visible()
    parser.feed(path.read_text(encoding="utf-8"))
    failures: list[str] = []

    visible = "".join(parser.prose) + "".join(parser.code_text)
    for line_no, line in enumerate(visible.splitlines(), 1):
        for char, name in FORBIDDEN.items():
            if char in line:
                failures.append(f"{name} in visible text: {line.strip()!r}")

    words = len("".join(parser.prose).split())
    if words >= WORD_LIMIT:
        failures.append(f"prose is {words} words, limit is under {WORD_LIMIT}")

    headline = " ".join("".join(parser.h1).split())
    if headline != HEADLINE:
        failures.append(f"headline is {headline!r}, expected {HEADLINE!r}")

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print(f"ok: no dashes, {words} words of prose, headline exact")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("site/index.html")
    sys.exit(main(target))
