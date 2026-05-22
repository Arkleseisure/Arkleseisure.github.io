#!/usr/bin/env python3
"""Estimate AI-risk worldviews for a list of public figures by shelling out to
`claude -p`. Writes JSON to `worldviews-generated.json` next to this file for
review before merging into data.js.

Usage:
  python3 generate-worldviews.py            # do everyone in parallel
  python3 generate-worldviews.py yudkowsky  # do just one (or several) by slug
  python3 generate-worldviews.py --force    # redo even if already saved
  python3 generate-worldviews.py -j 8       # set worker count (default 5)
"""
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Leaves of the ai-risk-by-type tree. Keep in sync with data.js.
LEAVES = [
    ("t-ai-inc",       "P(AI development raises P(existential catastrophe) within 30y, vs counterfactual where AI research plateaued before meaningfully affecting it — i.e. is AI net-bad for x-risk)"),
    ("t-d-no-inc",     "P(existential catastrophe occurs within 30y | AI doesn't raise its probability) — base rate from non-AI causes"),
    ("t-single",       "P(the danger comes from a single dominant AI rather than a multipolar landscape | AI raises P(D))"),
    ("t-d-multi",      "P(existential catastrophe occurs | the AI risk is multipolar, i.e. several AIs / no single decisive one)"),
    ("t-has-rep",      "P(the dominant AI has an internal model of D as a concept (vs raises P(D) via reward hacking / side effects without representing the danger) | a single dominant AI is what raises P(D))"),
    ("t-d-no-rep",     "P(existential catastrophe occurs | dominant AI raises P(D) but has no internal model of D — 'unaware harm')"),
    ("t-expects",      "P(the AI expects D to become more likely / intends or foresees D | AI has an internal model of D)"),
    ("t-d-expects",    "P(existential catastrophe occurs | the AI expects D — deliberate / foreseen case)"),
    ("t-d-no-expects", "P(existential catastrophe occurs | AI has internal model of D but does not expect it — miscalculation / wrong beliefs)"),
]

# Public figures + framing. Sean is doing his own; not auto-generated.
#
# Every bio ends with a "30-year framing:" line that translates the person's
# stated headline number to specifically the 30-year horizon used by the tree.
# For fast-takeoff doomers (Yudkowsky, Tegmark, Kokotajlo) the 30-year number
# is essentially the lifetime number — they expect the bulk of risk soon.
# For thinkers who reason over centuries (Ord), most of the risk is later,
# so the 30-year value is substantially smaller. For timeframe-insensitive
# skeptics (LeCun, Ng) the value is the same regardless of horizon.
PEOPLE = [
    ("yampolskiy", "Roman Yampolskiy",   "AI safety researcher at University of Louisville. Argues that AI alignment is provably impossible in principle (cf. his papers on uncontrollability of superintelligence). Publicly estimates P(doom) at 99.9%–99.999999% within 100 years — even more extreme than Yudkowsky. Treats every step from current AI to superintelligence as compounding uncontrollable risk. 30-year framing: he expects transformative AI well within decades, so 30y P(doom) ~95–99% — slightly lower than his 100-year value but still essentially saturated."),
    ("tegmark",    "Max Tegmark",        "MIT physicist, co-founder of the Future of Life Institute. Organised the 2023 'pause' open letter. Publicly estimates P(doom) above 90%. Argues humanity is sleepwalking into an uncontrolled intelligence explosion; treats the gap between AI capability growth and alignment progress as the proximate cause of catastrophe. 30-year framing: his 90%+ headline is implicitly near-term (he expects an intelligence explosion soon), so 30y P(doom) ~85–95%."),
    ("kokotajlo",  "Daniel Kokotajlo",   "Former OpenAI alignment researcher, co-author of the 'AI 2027' scenario. Publicly states P(doom) at ~70–80% within the next few decades. Reasons primarily via concrete operational scenarios about how frontier AI labs lose control during rapid takeoff, racing dynamics, and inadequate safety testing. 30-year framing: his headline number is already aimed at the next-few-decades horizon, so 30y P(doom) ~70–80% unchanged."),
    ("zvi",        "Zvi Mowshowitz",     "Independent AI writer at 'Don't Worry About the Vase'. The most-read synthesizer of the rationalist/EA AI-safety community. Publicly states P(doom) at ~70%. Treats alignment as deeply unsolved, race dynamics as severe, and humanity as unlikely to coordinate in time; reasonably representative of the broader high-concern rationalist cluster. 30-year framing: his ~70% is implicitly near-term (rationalist consensus expects AGI within 10–20y), so 30y P(doom) ~65–75% essentially unchanged."),
    ("yudkowsky",  "Eliezer Yudkowsky",  "Founder of MIRI. Long-standing AI doomer; publicly states P(doom) at >95% (often phrased as 'near-certain' or 99%). Argues current AI trajectory leads near-certainly to extinction absent dramatic intervention. Treats instrumental convergence as near-certain: any sufficiently capable goal-directed AI will develop a rich world-model including humans, and will strategically plan a 'treacherous turn' because removing oversight is instrumentally convergent. 30-year framing: he expects fast takeoff well within 30y, so 30y P(doom) ~90–98% — essentially unchanged from his headline."),
    ("lecun",      "Yann LeCun",         "Meta Chief AI Scientist. Publicly estimates P(doom) below 0.01% — essentially zero. Strongly skeptical that current LLM-based AI is on the path to AGI or x-risk; argues future systems will be controllable by design (his JEPA / objective-driven AI agenda). Treats x-risk concerns as fundamentally misplaced. 30-year framing: his estimate is timeframe-insensitive (the disagreement is about whether AI is dangerous at all), so 30y P(doom) ~0.01–0.1% unchanged."),
    ("hinton",     "Geoffrey Hinton",    "Deep-learning pioneer. After leaving Google in 2023 publicly raised existential-risk concerns about AI; estimates 10–20% P(extinction) all-things-considered, with a separate 'independent impression' closer to 50%. 30-year framing: his all-things-considered ~15% is over a long horizon (his comments on '30y to extinction' suggest he sees substantial mass in that window but also beyond), so 30y P(doom) ~8–15%."),
    ("christiano", "Paul Christiano",    "Founder of ARC, pioneer of RLHF. Publicly estimates ~46% total P(existential catastrophe) — combining ~10–20% from misaligned AI takeover and another ~25–30% from other AI-related failure modes (gradual misalignment, structural failures, misuse). His ~50% figure is the all-encompassing total, not just takeover. 30-year framing: his ~46% is roughly his lifetime/century estimate; he models transformative AI as likely-but-not-certain in 30y, so 30y P(doom) ~30–40%."),
    ("amodei",     "Dario Amodei",       "CEO of Anthropic. Publicly states ~10–25% P(doom). Believes powerful AI is arriving soon and brings real existential-level risks, but is broadly optimistic about navigating them with careful development. 30-year framing: his ~10–25% is explicitly about powerful AI 'arriving soon', so already 30y-aligned. 30y P(doom) ~10–25% unchanged."),
    ("toner",      "Helen Toner",        "AI policy researcher, former OpenAI board member. Focuses on governance, oversight, and policy responses; concerned about catastrophic risks especially from race dynamics and weak governance. No published numeric estimate; reasonable inference ~10–20%. 30-year framing: governance risks compound over time but the most acute window is the next decade or two, so 30y P(doom) ~8–18%."),
    ("bengio",     "Yoshua Bengio",      "Turing Award laureate, founder of MILA. Since 2023 actively focused on AI safety, especially loss-of-control risks from agentic systems. Publicly states ~20% P(doom); advocates moratoria on certain research. 30-year framing: his ~20% is over the period of transformative AI emergence, which he sees as imminent, so 30y P(doom) ~15–22% essentially unchanged."),
    ("russell",    "Stuart Russell",     "UC Berkeley, author of Human Compatible. Argues the standard AI paradigm of optimizing fixed objectives is structurally unsafe; advocates 'provably beneficial AI'. Public estimate ~20% P(doom). Concerned but solution-oriented. 30-year framing: his ~20% is the all-encompassing concern about the current paradigm; assuming the standard paradigm dominates the next 30y (likely), 30y P(doom) ~15–22%."),
    ("cotra",      "Ajeya Cotra",        "Open Philanthropy. Author of the bio anchors forecast. Empirical, careful; moderate-but-concerned position, with ~15–25% range on transformative-AI doom. 30-year framing: her TAI forecast has substantial probability mass in the next 30y, and her doom estimate is conditional on TAI arriving, so 30y P(doom) ~12–22% (slightly under her unconditional figure)."),
    ("ord",        "Toby Ord",           "Author of The Precipice. Frames AI alongside other x-risks; estimated ~10% from AI this century (one of the highest in his catalogue, but lower than the doomer cluster). 30-year framing: his explicit horizon is 100 years; he treats risk as roughly uniformly distributed but slightly back-loaded (capabilities accelerating), so 30y P(doom) ~3–5%."),
    ("karnofsky",  "Holden Karnofsky",   "Co-founder of Open Philanthropy. Public estimate ~50% P(doom), combining misaligned takeover with other AI-driven catastrophic failure modes. Thoughtful empirical reasoner; treats transformative AI within decades as the dominant existential risk this century. 30-year framing: his ~50% is more century-flavored than decade-flavored — he treats transformative AI as 'within decades' but his risk model spans the long emergence; 30y P(doom) ~30–40%."),
    ("hassabis",   "Demis Hassabis",     "CEO of Google DeepMind. Publicly declines to give a precise P(doom) number, calling it 'non-zero and probably non-negligible'. Signed the CAIS extinction-risk statement. Builds frontier systems while emphasising scientific benefits alongside safety. Reasonable inference ~10–20%. 30-year framing: he builds frontier systems and expects AGI within ~10y, so any P(doom) he holds is essentially aimed at the 30y window. 30y P(doom) ~8–18%."),
    ("marcus",     "Gary Marcus",        "AI capabilities skeptic. Argues current LLMs are not on the path to AGI and that scaling alone won't produce existential-risk-level systems; says 'extinction is pretty unlikely'. Has stated low P(extinction), with most of his probability mass on dystopia/catastrophe-short-of-extinction rather than human extinction. 30-year framing: his low number is because he doesn't expect existential-risk-capable systems within decades; 30y P(doom) ~1–4% essentially unchanged."),
    ("ng",         "Andrew Ng",          "Coursera / Stanford. AI optimist; has explicitly dismissed near-term existential AI risk concerns as 'overwhelmingly unlikely' and akin to 'worrying about overpopulation on Mars'. 30-year framing: he explicitly says 'near-term', so his estimate is already 30y-aimed. 30y P(doom) ~0.5–2% unchanged."),
]

LEAF_LINES = "\n".join(f"- {lid}: {desc}" for lid, desc in LEAVES)

PROMPT = """You are estimating what {name} would plausibly say about a structured AI-risk question, based on their public statements and known views.

Bio framing:
{bio}

Below is the leaf parameters of an assumption tree decomposing P(existential catastrophe from AI within 30 years). Each leaf is a probability, conditional on its parent context.

{leaves}

IMPORTANT — horizon adjustment: the tree is specifically conditioned on the next 30 years. The bio's "30-year framing:" line is the calibrated headline number to anchor on, NOT the raw public-statement figure. The marginal-probability leaves (t-ai-inc, t-d-no-inc) are the most timeline-sensitive; the conditional leaves (t-d-multi, t-d-no-rep, t-d-expects, t-d-no-expects, t-single, t-has-rep, t-expects) describe properties of worlds and AI types conditional on the relevant configuration already obtaining, so they are largely timeline-insensitive and should reflect the person's underlying model rather than their headline probability.

For each leaf, estimate {name}'s best-guess point probability, a 90% credible interval (lo, hi), AND a one- or two-sentence rationale that cites the specific public views or arguments that drive the choice (so a reader can sanity-check it). When the rationale touches the horizon (mostly for t-ai-inc and t-d-no-inc), make the 30-year adjustment explicit.

Output strictly valid JSON in <json>...</json> tags with the structure below. Do not include anything before or after the JSON tags.

<json>
{{
  "name": "{name}",
  "perspective": "inside",
  "description": "one short sentence describing how this view differs from the median",
  "probabilities": {{
    "t-ai-inc": 0.00,
    "t-d-no-inc": 0.00,
    "t-single": 0.00,
    "t-d-multi": 0.00,
    "t-has-rep": 0.00,
    "t-d-no-rep": 0.00,
    "t-expects": 0.00,
    "t-d-expects": 0.00,
    "t-d-no-expects": 0.00
  }},
  "ranges": {{
    "t-ai-inc":       {{ "lo": 0.00, "hi": 0.00 }},
    "t-d-no-inc":     {{ "lo": 0.00, "hi": 0.00 }},
    "t-single":       {{ "lo": 0.00, "hi": 0.00 }},
    "t-d-multi":      {{ "lo": 0.00, "hi": 0.00 }},
    "t-has-rep":      {{ "lo": 0.00, "hi": 0.00 }},
    "t-d-no-rep":     {{ "lo": 0.00, "hi": 0.00 }},
    "t-expects":      {{ "lo": 0.00, "hi": 0.00 }},
    "t-d-expects":    {{ "lo": 0.00, "hi": 0.00 }},
    "t-d-no-expects": {{ "lo": 0.00, "hi": 0.00 }}
  }},
  "reasoning": {{
    "t-ai-inc":       "one- or two-sentence rationale citing public views",
    "t-d-no-inc":     "...",
    "t-single":       "...",
    "t-d-multi":      "...",
    "t-has-rep":      "...",
    "t-d-no-rep":     "...",
    "t-expects":      "...",
    "t-d-expects":    "...",
    "t-d-no-expects": "..."
  }}
}}
</json>"""


def query_claude(prompt: str, model: str = "sonnet") -> str:
    """Invoke `claude -p` using Max-plan OAuth.

    Following the inkhaven-blogger-rankings convention: unset ANTHROPIC_API_KEY
    so `claude -p` falls through to OAuth (Max plan) auth instead of preferring
    the API key (which would bill against API credits). Pass the prompt via
    stdin and ask for JSON output so we can parse the actual response cleanly.
    """
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    cmd = ["claude", "-p", "--model", model, "--output-format", "json"]
    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude exited {result.returncode}: {result.stderr.strip()[-300:]}")

    # `--output-format json` wraps the model response in a JSON envelope:
    # { "result": "...model text...", "is_error": bool, ... }
    try:
        parsed = json.loads(result.stdout)
    except Exception:
        # Fallback: treat the whole stdout as the response text.
        return result.stdout
    if parsed.get("is_error"):
        raise RuntimeError(f"claude API error: {parsed.get('result')}")
    return parsed.get("result", "") or result.stdout


def extract_json(text: str) -> dict:
    m = re.search(r"<json>\s*(\{[\s\S]*?\})\s*</json>", text)
    if not m:
        m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if not m:
        m = re.search(r"(\{[\s\S]*\})", text)
    if not m:
        raise ValueError(f"No JSON in response:\n{text[:500]}")
    return json.loads(m.group(1))


def run_one(slug: str, name: str, bio: str) -> tuple[str, dict]:
    """Query Claude for one person; return (slug, result_dict)."""
    prompt = PROMPT.format(name=name, bio=bio, leaves=LEAF_LINES)
    try:
        response = query_claude(prompt)
        parsed = extract_json(response)
        parsed["slug"] = slug
        return slug, parsed
    except Exception as e:
        return slug, {"slug": slug, "name": name, "error": str(e)}


def main():
    args = sys.argv[1:]
    force = "--force" in args
    workers = 5
    if "-j" in args:
        i = args.index("-j")
        workers = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    slugs = [a for a in args if a != "--force"]
    targets = [p for p in PEOPLE if not slugs or p[0] in slugs]

    out_path = Path(__file__).resolve().parent / "worldviews-generated.json"
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    todo = []
    for slug, name, bio in targets:
        if slug in existing and "error" not in existing[slug] and not force:
            print(f"= {name}: already done (--force to redo)", flush=True)
            continue
        todo.append((slug, name, bio))

    if not todo:
        print("Nothing to do.")
        return

    print(f"Querying {len(todo)} people with {workers} parallel workers...", flush=True)
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, slug, name, bio): name for slug, name, bio in todo}
        for fut in as_completed(futures):
            who = futures[fut]
            slug, result = fut.result()
            with write_lock:
                existing[slug] = result
                out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
            if "error" in result:
                print(f"  ERROR  {who}: {result['error']}", flush=True)
            else:
                note = (result.get("description") or "")[:90]
                print(f"  done   {who}: {note}", flush=True)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
