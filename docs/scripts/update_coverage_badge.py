#!/usr/bin/env python3
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def badge_color(percent: float) -> str:
    if percent < 60:
        return "#e05d44"
    if percent < 80:
        return "#dfb317"
    return "#4c1"


def make_svg(label: str, value: str, color: str) -> str:
    # Simple static badge, no external service needed.
    left_width = 68
    right_width = 68
    total_width = left_width + right_width

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="20" role="img" aria-label="{label}: {value}">
  <title>{label}: {value}</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <rect rx="3" width="{left_width}" height="20" fill="#555"/>
  <rect rx="3" x="{left_width}" width="{right_width}" height="20" fill="{color}"/>
  <rect rx="3" width="{total_width}" height="20" fill="url(#s)"/>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">
    <text x="{left_width / 2}" y="14">{label}</text>
    <text x="{left_width + right_width / 2}" y="14">{value}</text>
  </g>
</svg>
"""


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: update_coverage_badge.py <coverage.xml> <output.svg>", file=sys.stderr)
        return 2

    coverage_xml = Path(sys.argv[1])
    output_svg = Path(sys.argv[2])

    root = ET.parse(coverage_xml).getroot()
    percent = float(root.attrib["line-rate"]) * 100.0
    percent_text = f"{percent:.1f}%"
    color = badge_color(percent)

    output_svg.parent.mkdir(parents=True, exist_ok=True)
    output_svg.write_text(make_svg("coverage", percent_text, color), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
