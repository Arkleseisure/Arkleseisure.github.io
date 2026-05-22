#!/usr/bin/env python3
"""Fetch Sean's LW Inkhaven posts and emit local HTML using the site template.

Picks the verbatim postContent div out of LW's SSR HTML, strips LW's chrome,
wraps in the standard site post template.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

POSTS = [
    # (slug, section, lw_url, title_override_or_none, date_iso, date_display, read_time_min)
    (
        "the-quest-for-general-intelligence-is-hitting-a-wall",
        "safety",
        "https://www.lesswrong.com/posts/xZsuBaQFGEb743RiM/the-quest-for-general-intelligence-is-hitting-a-wall",
        "The quest for general intelligence is hitting a wall [April Fool's]",
        "2026-04-01",
        "April 1, 2026",
        3,
    ),
    (
        "changes-to-an-optimised-thing-make-it-worse",
        "life",
        "https://www.lesswrong.com/posts/YfeZzj5CuEeTwyyNS/changes-to-an-optimised-thing-make-it-worse",
        "Changes to an optimised thing make it worse",
        "2026-04-04",
        "April 4, 2026",
        4,
    ),
    (
        "estimates-of-the-expected-utility-gain-of-ai-safety-research",
        "safety",
        "https://www.lesswrong.com/posts/gXYeWoAfSrdGogchp/estimates-of-the-expected-utility-gain-of-ai-safety-research",
        "Estimates of the expected utility gain of AI Safety Research",
        "2026-04-05",
        "April 5, 2026",
        3,
    ),
    (
        "most-people-cant-juggle-one-ball",
        "universe",
        "https://www.lesswrong.com/posts/jTGbKKGqs5EdyYoRc/most-people-can-t-juggle-one-ball",
        "Most people can't juggle one ball",
        "2026-04-07",
        "April 7, 2026",
        12,
    ),
    (
        "stockfish-is-not-a-chess-superintelligence",
        "safety",
        "https://www.lesswrong.com/posts/ETxKteTat4KfREeBu/generalisation-isn-t-actually-that-important",
        # Title was changed post-publication on LW
        "Stockfish is not a chess superintelligence (and it doesn't need to be)",
        "2026-04-08",
        "April 8, 2026",
        5,
    ),
    (
        "ai-identity-is-not-tied-to-its-model",
        "safety",
        "https://www.lesswrong.com/posts/4qngqERKakrCJpdEe/ai-identity-is-not-tied-to-its-model",
        "AI identity is not tied to its model",
        "2026-04-09",
        "April 9, 2026",
        5,
    ),
    (
        "some-ai-threats-people-arent-thinking-about",
        "safety",
        "https://www.lesswrong.com/posts/BrEEJQwP2BG5ZE3kk/some-ai-threats-people-aren-t-thinking-about",
        "Some AI threats people aren't thinking about",
        "2026-04-13",
        "April 13, 2026",
        7,
    ),
]


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8")


def extract_post_content(html: str) -> str:
    """Pull the innerHTML of <div id="postContent"> using a balanced-div walk."""
    marker = '<div id="postContent">'
    start = html.find(marker)
    if start < 0:
        raise RuntimeError("postContent div not found")
    pos = start + len(marker)
    depth = 1
    i = pos
    while i < len(html) and depth > 0:
        open_idx = html.find("<div", i)
        close_idx = html.find("</div>", i)
        if close_idx < 0:
            break
        if 0 <= open_idx < close_idx:
            depth += 1
            i = open_idx + 4
        else:
            depth -= 1
            i = close_idx + len("</div>")
    # Slice off the outer <div id="postContent">...</div>
    inner = html[start + len(marker) : i - len("</div>")]
    return inner


def clean_lw_html(inner: str) -> str:
    """Strip LW-specific wrappers and rewrite a few class hooks to plain HTML."""
    # LW gives every block a numeric id (block0, block1...). Drop them.
    inner = re.sub(r' id="block\d+"', "", inner)
    # Drop heading anchor ids but keep footnote ids (need them for the
    # internal back-links between footnote refs and footnote items).
    inner = re.sub(
        r' id="(?!fn(?:ref)?[A-Za-z0-9_-]+)[A-Za-z0-9_-]+"', "", inner
    )
    # Drop noise classes
    inner = re.sub(r' class="LinkStyles-link"', "", inner)
    inner = re.sub(r' rel="noreferrer"', "", inner)
    # LW renders internal links as /posts/...; rewrite to absolute LW URLs.
    inner = re.sub(
        r'href="(/posts/[^"]+)"', r'href="https://www.lesswrong.com\1"', inner
    )
    # <li value="N"> is auto-generated; drop the value attribute.
    inner = re.sub(r'<li value="\d+">', "<li>", inner)
    # Unwrap <span> wrappers that LW puts around text runs and around links.
    # Repeated passes handle nested <span><span>...</span></span> cases.
    for _ in range(6):
        new = re.sub(r"<span>([^<]*?)</span>", r"\1", inner)
        new = re.sub(
            r"<span>(<a [^>]*>[^<]*</a>)</span>", r"\1", new, flags=re.DOTALL
        )
        if new == inner:
            break
        inner = new
    # <i> </i> (whitespace-only emphasis from LW's editor) → just the space.
    inner = re.sub(r"<i>(\s+)</i>", r"\1", inner)
    inner = re.sub(r"<em>(\s+)</em>", r"\1", inner)
    # Strip the redundant top-level <div><div>...</div></div> wrappers.
    inner = re.sub(r"^\s*<div>\s*<div>\s*", "", inner)
    inner = re.sub(r"\s*</div>\s*</div>\s*$", "", inner)
    # HTML entities back to normal characters.
    inner = (inner.replace("&#x27;", "'")
                  .replace("&quot;", '"')
                  .replace("&amp;", "&")
                  .replace("&#x2F;", "/"))
    return inner.strip()


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="icon" type="image/svg+xml" href="/favicon.svg">
  <title>{title} — Life, The Universe & That Safety Thing</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&family=Source+Sans+3:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="../../css/style.css">{mathjax}
</head>
<body>
  <header class="site-header">
    <div class="container">
      <div class="header-inner">
        <a href="../../index.html" class="home-link">Home</a>
        <nav class="main-nav" role="navigation" aria-label="Main navigation">
          <div class="nav-item">
            <a href="../../life.html" class="nav-link{life_active}" data-section="life">Life</a>
          </div>
          <div class="nav-item">
            <a href="../../universe.html" class="nav-link{universe_active}" data-section="universe">The Universe</a>
          </div>
          <div class="nav-item">
            <a href="../../safety.html" class="nav-link{safety_active}" data-section="safety">AI Safety</a>
            <div class="dropdown">
              <a href="../../safety.html" class="dropdown-link">Articles</a>
              <a href="../../projects.html" class="dropdown-link">Projects</a>
            </div>
          </div>
          <div class="nav-item">
            <a href="../../about.html" class="nav-link">About</a>
          </div>
        </nav>
        <button class="nav-toggle" aria-label="Toggle navigation" aria-expanded="false">
          <span></span><span></span><span></span>
        </button>
      </div>
    </div>
  </header>

  <article class="article">
    <div class="container content-narrow">
      <header class="article-header">
        <span class="section-label {section}">{section_label}</span>
        <h1>{title}</h1>
        <p class="article-meta">
          <time datetime="{date_iso}">{date_display}</time> &middot; {read_time} min read &middot;
          <a href="{lw_url}" target="_blank" rel="noopener">Originally on LessWrong</a>
        </p>
      </header>

      <div class="article-content">
{body}
      </div>
    </div>
  </article>

  <footer class="site-footer">
    <div class="container">
      <div class="footer-content">
        <div class="footer-links">
          <a href="../../about.html">About</a>
          <a href="https://github.com/Arkleseisure" target="_blank" rel="noopener">GitHub</a>
        </div>
        <p>&copy; 2024-2026 Sean Herrington. All rights reserved.</p>
      </div>
    </div>
  </footer>

  <script src="../../js/main.js"></script>
</body>
</html>
"""

SECTION_LABEL = {
    "safety": "AI Safety",
    "life": "Life",
    "universe": "The Universe",
}


MATHJAX_TAG = (
    '\n  <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" '
    'async></script>'
)


def build_post(post):
    slug, section, lw_url, title, date_iso, date_display, read_time = post
    raw = fetch(lw_url)
    inner = extract_post_content(raw)
    body = clean_lw_html(inner)
    # Only load MathJax on posts that contain LW's inline math markers.
    mathjax = MATHJAX_TAG if "mjx-container" in body or "MathJax" in body else ""
    out = TEMPLATE.format(
        title=title,
        section=section,
        section_label=SECTION_LABEL[section],
        date_iso=date_iso,
        date_display=date_display,
        read_time=read_time,
        body=body,
        lw_url=lw_url,
        mathjax=mathjax,
        life_active=" active" if section == "life" else "",
        universe_active=" active" if section == "universe" else "",
        safety_active=" active" if section == "safety" else "",
    )
    out_path = ROOT / "posts" / section / f"{slug}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out, encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)} ({len(body)} chars)")


if __name__ == "__main__":
    for p in POSTS:
        try:
            build_post(p)
        except Exception as e:
            print(f"FAILED {p[0]}: {e}", file=sys.stderr)
