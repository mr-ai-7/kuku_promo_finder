"""Promo Finder v2 — transparent, show-independent screenplay clip ranking.

Offline script-only baseline, not a video editor or predictive ad model.
The system requires timestamped scripts; no show names or character names are hard-coded.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TIME = r"\d{2,3}:\d{2}\.\d{2}"
EVENT_RE = re.compile(
    rf"^\*\*(?P<label>[^\n]+?):\*\*\s*\nStart:\s*\((?P<start>{TIME})\)\s*\|\s*End:\s*\((?P<end>{TIME})\)\s*\n"
    r"(?P<body>.*?)(?=\n\s*\n|\n\s*\*\*|\nSCENE\s+\d+|\n--- SCRIPT END ---|\Z)",
    re.M | re.S | re.I,
)
EP_FILE_RE = re.compile(r"episode[-_ ]?(\d+)", re.I)
SCENE_RE = re.compile(r"^SCENE\s+\d+\b", re.I | re.M)
SCENE_START_RE = re.compile(rf"^Scene start:\s*({TIME})", re.I | re.M)
SCENE_END_RE = re.compile(rf"^Scene end:\s*({TIME})", re.I | re.M)

# General screenplay signals, not tied to royal-family dramas or show characters.
DRAMA = re.compile(
    r"\b(betray(?:ed|al)?|secret|arrest(?:ed)?|escape|evidence|stole|steal|stolen|"
    r"lie(?:d|s)?|lying|fraud|poison(?:ed)?|danger|threat(?:en)?|"
    r"kill(?:ed)?|dead|death|dying|disappear(?:ed)?|missing|trapped|"
    r"fire|destroy(?:ed)?|forged|conspiracy|traitor|guilty|innocent|"
    r"save|rescue|discovered|proof|abandon(?:ed)?|revenge|expose|"
    r"shame|disgrace|ruin|risk|truth|hide|hidden|found out|"
    r"can't breathe|cannot breathe|impossible|never knew)\b", re.I
)
CONFRONTATION = re.compile(
    r"\b(you (?:lied|stole|betrayed|killed|framed|hid|cheated|can't|cannot|will not)|"
    r"i (?:refuse|won't|will never)|how dare you|stop right there|"
    r"you are under arrest|you've been accused|i saw you|"
    r"don't (?:touch|move|lie)|let (?:her|him|them|me) go)\b", re.I
)
REVELATION = re.compile(
    r"\b(it was (?:you|me|him|her|them)|the truth is|i am (?:your|the)|"
    r"i'm (?:your|the)|i know (?:what|who|why)|nobody knows|"
    r"you are (?:not|the)|the real|that's not true|"
    r"this is (?:the|your)|you never knew|i found (?:it|the)|"
    r"i have proof|the proof is|i can prove)\b", re.I
)
URGENT = re.compile(r"\b(stop|wait|run|help|look out|don't|no!|listen|please|hurry|hold on)\b", re.I)
ACTION_SHOCK = re.compile(
    r"\b(slams?|bursts?|crashes?|collapses?|grabs?|snatches?|rushes?|"
    r"storms?|reveals?|exposes?|pulls? (?:out|open)|drops?|shatters?|"
    r"discovers?|gasps?|freezes?|unlocks?|opens? the door|"
    r"appears?|steps? into view|suddenly|abruptly|explodes?|"
    r"points? at|falls? to (?:the|his|her) knees)\b", re.I
)
THREAT_FUTURE = re.compile(r"\b(we settle|you will pay|you will regret|you won't escape|you cannot escape|you will never|i will expose|or else|i will stop|you are next|i will find|what happens next)\b", re.I)
WH_QUESTION = re.compile(r"\b(who|what|why|where|when|how|which)\b[^?]{0,140}\?", re.I)
RHETORICAL = re.compile(
    r"\b(how dare you|who cares|what did you expect|what else could i|"
    r"isn't that obvious|did you really think|are you kidding|"
    r"what's wrong with you|what do you take me for)\b", re.I
)
INCOMPLETE_START = re.compile(
    r"^(?:\.\.\.|…|,|[—-]\s+|and\b|but\b|because\b|although\b|"
    r"while\b|which\b|whereas\b|so that\b|except that\b|"
    r"even though\b|if only\b)", re.I
)
RESOLVED = re.compile(
    r"\b(we (?:won|are safe|did it)|it's (?:over|settled|finished)|"
    r"it is (?:over|settled)|thank goodness|safe now|"
    r"finally (?:safe|free)|everything (?:is|will be) (?:fine|okay)|"
    r"i forgive you|i accept|you are free|problem solved|"
    r"the danger is over|we saved (?:him|her|them)|he (?:is|was) saved|"
    r"she (?:is|was) saved)\b", re.I
)
EXPOSITION = re.compile(
    r"\b(as you know|let me explain|first of all|"
    r"i will explain|years ago|according to the records|"
    r"let me tell you the whole story|the history of|"
    r"in the old days)\b", re.I
)
RELATIONSHIP = re.compile(r"\b(my|your|his|her|our)\s+(?:mother|father|brother|sister|wife|husband|son|daughter|partner|friend|boss|child)\b", re.I)
SENSITIVE = re.compile(r"\b(naked|undress|intimate|seductive|gore|execution|strangled|corpse|bloodied)\b", re.I)


def ts_to_s(t: str) -> float:
    m, s = t.split(":")
    seconds = int(m) * 60 + float(s)
    if float(s) >= 60:
        raise ValueError(f"Invalid seconds in timestamp: {t}")
    return round(seconds, 2)


def s_to_ts(t: float) -> str:
    n = int(round(t * 100))
    return f"{n // 6000:02d}:{(n % 6000) // 100:02d}.{n % 100:02d}"


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def dialogue_text(value: str) -> str:
    return clean(value).strip('"“” \t\n')


@dataclass(frozen=True)
class Event:
    kind: str
    label: str
    start: float
    end: float
    text: str


@dataclass
class Episode:
    number: int
    name: str
    logline: str
    continuity: str
    events: list[Event]
    total_s: float
    scene_count: int
    scenes: list[tuple[float, float]]
    warnings: list[str]


def read_inputs(input_path: str | Path) -> list[tuple[str, str]]:
    """Read files in-memory without extracting ZIP paths to disk."""
    p = Path(input_path)
    files: list[tuple[str, str]] = []
    max_items, max_each, max_all = 300, 3_000_000, 80_000_000
    if p.is_file() and zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            infos = [i for i in z.infolist() if i.filename.lower().endswith(".txt")
                     and not i.filename.startswith("__MACOSX/")
                     and not Path(i.filename).name.startswith("._")]
            if len(infos) > max_items or sum(i.file_size for i in infos) > max_all:
                raise ValueError("Input pack is too large (max 300 scripts / 80 MB uncompressed).")
            for info in infos:
                if info.file_size > max_each:
                    raise ValueError(f"Script too large: {info.filename}")
                files.append((info.filename, z.read(info).decode("utf-8-sig", errors="replace")))
    elif p.is_dir():
        paths = sorted(f for f in p.rglob("*.txt") if f.is_file())
        if len(paths) > max_items or sum(f.stat().st_size for f in paths) > max_all:
            raise ValueError("Input folder is too large.")
        for f in paths:
            if f.stat().st_size > max_each:
                raise ValueError(f"Script too large: {f.name}")
            files.append((str(f.relative_to(p)), f.read_text(encoding="utf-8-sig", errors="replace")))
    elif p.is_file() and p.suffix.lower() == ".txt":
        files.append((p.name, p.read_text(encoding="utf-8-sig", errors="replace")))
    else:
        raise ValueError("Upload a ZIP containing timestamped .txt scripts, a folder, or a single .txt script.")
    if not files:
        raise ValueError("No .txt screenplay files were found.")
    return sorted(files, key=lambda x: x[0])


def parse_episode(name: str, txt: str, fallback: int) -> Episode:
    matches = list(EVENT_RE.finditer(txt))
    events: list[Event] = []
    warnings: list[str] = []
    for m in matches:
        label = clean(m.group("label"))
        kind = label.upper() if label.lower() in {"camera", "action", "sound fx", "music", "voiceover"} else "DIALOGUE"
        try:
            a, b = ts_to_s(m.group("start")), ts_to_s(m.group("end"))
        except ValueError as e:
            warnings.append(str(e)); continue
        if b <= a:
            warnings.append(f"Non-positive duration for {label}")
            continue
        body = dialogue_text(m.group("body"))
        if body:
            events.append(Event(kind, label, a, b, body))
    ep_header = re.search(r"^EPISODE NUMBER:\s*Episode\s*(\d+)", txt, re.I | re.M)
    filename_match = EP_FILE_RE.search(Path(name).name)
    num = int(ep_header.group(1) if ep_header else filename_match.group(1) if filename_match else fallback)
    logline = re.search(r"^EPISODE LOGLINE:\s*(.+)", txt, re.I | re.M)
    continuity = re.search(r"^CONTINUITY NOTES:\s*(.+)", txt, re.I | re.M)
    starts = [ts_to_s(t) for t in SCENE_START_RE.findall(txt)]
    ends = [ts_to_s(t) for t in SCENE_END_RE.findall(txt)]
    scenes = [(a, b) for a, b in zip(starts, ends) if b > a]
    events.sort(key=lambda e: (e.start, e.end))
    total = max([e.end for e in events] + ends + [0])
    if len(starts) != len(ends):
        warnings.append("Unmatched scene start/end markers")
    if not events:
        warnings.append("No supported timed events were parsed")
    if sum(e.kind == "DIALOGUE" for e in events) < 3:
        warnings.append("Fewer than three timed dialogue events")
    return Episode(num, name, logline.group(1).strip() if logline else "",
                   continuity.group(1).strip() if continuity else "", events,
                   round(total, 2), len(SCENE_RE.findall(txt)), scenes, warnings)


def selected_dialogue(ep: Episode, start: float, end: float) -> list[Event]:
    return [e for e in ep.events if e.kind == "DIALOGUE" and e.start >= start - .02 and e.end <= end + .02]


def relevant_events(ep: Episode, start: float, end: float, kinds: set[str] | None = None) -> list[Event]:
    return [e for e in ep.events if e.end > start + .01 and e.start < end - .01
            and (not kinds or e.kind in kinds)]


def spans_dialogue(ep: Episode, timestamp: float) -> bool:
    return any(e.kind == "DIALOGUE" and e.start + .12 < timestamp < e.end - .12 for e in ep.events)


def candidate_boundaries(ep: Episode) -> tuple[list[float], list[float]]:
    start = {0.0}
    end = {round(ep.total_s, 2)}
    for e in ep.events:
        if e.kind == "DIALOGUE" or (e.kind == "ACTION" and len(e.text) > 18):
            start.add(round(e.start, 2)); end.add(round(e.end, 2))
    start.update(round(a, 2) for a, _ in ep.scenes)
    end.update(round(b, 2) for _, b in ep.scenes)
    return (sorted(t for t in start if t <= ep.total_s - 20 and not spans_dialogue(ep, t)),
            sorted(t for t in end if t >= 20 and not spans_dialogue(ep, t)))


def excerpt(e: Event, limit: int = 190) -> str:
    return e.text[:limit] + ("…" if len(e.text) > limit else "")


def scene_transitions(ep: Episode, a: float, b: float) -> int:
    # A segment may span a scene cut, but the editor needs a visible warning.
    return sum(a + .25 < scene_end < b - .25 for _, scene_end in ep.scenes)


def score_opening(e: Event, a: float) -> tuple[float, str]:
    t = e.text
    impact = 0.5
    reason = "immediate dramatic action" if e.kind == "ACTION" else "opening dialogue"
    if CONFRONTATION.search(t):
        impact += 2.0; reason = "direct confrontation or accusation"
    elif REVELATION.search(t):
        impact += 1.8; reason = "immediate revelation"
    elif WH_QUESTION.search(t):
        impact += 1.15; reason = "urgent opening question"
    if URGENT.search(t):
        impact += 1.1; reason = "urgent opening"
    if e.kind == "ACTION" and ACTION_SHOCK.search(t):
        impact += 1.6; reason = "visual disruption"
    if DRAMA.search(t):
        impact += .85
    if INCOMPLETE_START.search(t):
        impact -= 1.8
    if e.kind == "DIALOGUE" and len(t.split()) < 3:
        impact -= .8
    # Only the first three seconds qualify; earlier is generally stronger.
    delay = max(0, e.start - a)
    impact -= delay * .28
    return max(0, min(5, impact)), reason


def score_ending(e: Event, b: float, next_dialogue: Event | None) -> tuple[float, str]:
    t = e.text
    score = .55
    reason = "an unresolved end point"
    if t.rstrip().endswith("?"):
        score += 1.35; reason = "unanswered question at the cut"
        if WH_QUESTION.search(t):
            score += .7
        if RHETORICAL.search(t):
            score -= 1.9; reason = "possibly rhetorical question"
    if CONFRONTATION.search(t):
        score += 1.15; reason = "confrontation remains open"
    if REVELATION.search(t):
        score += .65; reason = "reveal immediately before the cut"
    if DRAMA.search(t):
        score += .7
    if THREAT_FUTURE.search(t):
        score += 1.15; reason = "high-stakes threat remains open"
    # A completed confident declaration is not automatically a cliffhanger.
    if (not t.rstrip().endswith("?") and e.kind == "DIALOGUE"
            and not THREAT_FUTURE.search(t) and not t.rstrip().endswith(("…", "...", "—"))):
        score -= 1.0
    if e.kind == "ACTION" and ACTION_SHOCK.search(t):
        score += 1.2; reason = "action suspended at the cut"
    if t.rstrip().endswith(("…", "...", "—")):
        score += .65
    if RESOLVED.search(t):
        score -= 2.25; reason = "conflict may already be resolved"
    if next_dialogue and 0 <= next_dialogue.start - b <= 6:
        # An immediate unanswered response is a plausible cut point, not proof of suspense.
        score += .4
    score -= max(0., b - e.end - .1) * .40
    return max(0, min(5, score)), reason


def score_candidate(ep: Episode, a: float, b: float) -> dict | None:
    duration = round(b - a, 2)
    if not (20 <= duration <= 60):
        return None
    ds = selected_dialogue(ep, a, b)
    if len(ds) < 3 or sum(len(e.text.split()) for e in ds) < 18:
        return None
    evs = relevant_events(ep, a, b, {"DIALOGUE", "ACTION"})
    early = [e for e in evs if e.start < a + 3 and e.end > a + .01]
    if not early:
        return None
    # Report the earliest observable meaningful beat; score its hook rather than
    # cherry-picking a dramatic line arriving after the first three seconds.
    opener = min(early, key=lambda e: (max(a, e.start), 0 if e.kind == "ACTION" else 1))
    hook, hook_reason = score_opening(opener, a)
    late = [e for e in evs if e.end > b - 3.3 and e.end <= b + .02]
    if not late:
        # A clip ending after silence cannot be rated as a strong cliffhanger.
        closer = max(evs, key=lambda e: e.end)
    else:
        closer = max(late, key=lambda e: (e.end, 1 if e.kind == "DIALOGUE" else 0))
    next_dialogue = next((e for e in ep.events if e.kind == "DIALOGUE" and e.start >= b - .01), None)
    cliff, cliff_reason = score_ending(closer, b, next_dialogue)
    combined = " ".join(e.text for e in ds)
    speakers = {re.sub(r"\s*\(.*", "", e.label).strip().lower() for e in ds}
    speaker_turns = sum(a1.label.split("(")[0] != a2.label.split("(")[0] for a1, a2 in zip(ds, ds[1:]))
    transitions = scene_transitions(ep, a, b)
    monologue = (max(Counter(e.label.split("(")[0] for e in ds).values()) / len(ds))
    # Standalone clarity proxies, not fabricated viewer comprehension labels.
    clarity = 2.75
    if len(speakers) >= 2: clarity += .65
    if len(ds) >= 6: clarity += .3
    if RELATIONSHIP.search(combined): clarity += .5
    if CONFRONTATION.search(combined): clarity += .3
    if INCOMPLETE_START.search(opener.text): clarity -= 1.6
    if opener.kind == "DIALOGUE" and re.match(r"^(?:he|she|they|it|this|that|those|him|her|them)\b", opener.text, re.I):
        clarity -= .55
    if monologue > .66 and len(ds) >= 6: clarity -= .8
    if EXPOSITION.search(combined): clarity -= .75
    if transitions: clarity -= min(1.5, .65 * transitions)
    clarity = round(max(0, min(5, clarity)), 2)
    urgency_signals = len(DRAMA.findall(combined)) + len(CONFRONTATION.findall(combined)) * 2
    conflict = min(5, .5 + urgency_signals * .38)
    pacing = min(5, (len(ds) / max(duration, 1)) * 14 + min(1.1, speaker_turns * .24))
    end_slice = " ".join(e.text for e in ds if e.start >= b - 12)
    resolved_late = bool(RESOLVED.search(end_slice))
    sensitive = bool(SENSITIVE.search(" ".join(e.text for e in evs)))
    exposition = bool(EXPOSITION.search(combined))
    warnings = []
    if hook < 1.75: warnings.append("Weak first-three-second hook: editor should review")
    if cliff < 1.7: warnings.append("Cliffhanger may not be unresolved or compelling")
    if INCOMPLETE_START.search(opener.text): warnings.append("Opening may begin mid-thought or with a dependent clause")
    if clarity < 2.8: warnings.append("Standalone clarity is uncertain: new-viewer review needed")
    if transitions: warnings.append(f"Crosses {transitions} scene boundary/boundaries: verify continuity")
    if resolved_late: warnings.append("Possible resolution near the end; suspense may be lost")
    if exposition: warnings.append("Contains explanatory dialogue; check promo pacing")
    if monologue > .66 and len(ds) >= 6: warnings.append("Long speaker-dominated exchange; may feel like exposition")
    if sensitive: warnings.append("Potentially sensitive content; editorial review required")
    if b - closer.end > 1.6: warnings.append("Last meaningful beat occurs before the clip ends")
    if opener.kind == "ACTION": warnings.append("Visual opening: verify footage matches script")
    if closer.kind == "ACTION": warnings.append("Visual cliffhanger: verify footage matches script")
    if RHETORICAL.search(closer.text) and closer.text.endswith("?"):
        warnings.append("Ending question may be rhetorical, not a cliffhanger")
    # Scores are interpretable relative rankings, NOT empirical engagement predictions.
    total = (hook / 5 * 31 + cliff / 5 * 32 + clarity / 5 * 21 + conflict / 5 * 10 + pacing / 5 * 6)
    total -= (5.5 * transitions + (8 if resolved_late else 0) + (4 if sensitive else 0))
    if hook < 1.2: total -= 6
    if cliff < 1.2: total -= 7
    total = round(max(0, min(100, total)), 2)
    reasons = [hook_reason, cliff_reason]
    if clarity >= 3.2: reasons.append("dialogue gives some standalone context")
    reason = "; ".join(dict.fromkeys(reasons)).capitalize() + "."
    return {
        "episode": ep.number,
        "segments": [{"episode": ep.number, "start": s_to_ts(a), "end": s_to_ts(b)}],
        "duration_seconds": duration,
        "score": total,
        "hook_score": round(hook, 2),
        "cliffhanger_score": round(cliff, 2),
        "self_contained_score": clarity,
        "conflict_score": round(conflict, 2),
        "pacing_score": round(pacing, 2),
        "hook": excerpt(opener),
        "hook_at": s_to_ts(max(a, opener.start)),
        "first_beat_delay_seconds": round(max(0, opener.start - a), 2),
        "cliffhanger": excerpt(closer),
        "cliffhanger_at": s_to_ts(closer.end),
        "why_it_works": reason,
        "quality_warnings": warnings,
        "scene_transitions": transitions,
        "dialogue_lines": len(ds),
        "speaker_count": len(speakers),
        "sensitive_flag": sensitive,
        "source_filename": Path(ep.name).name,
        "editorial_review": "Not reviewed",
    }


def generate_candidates(ep: Episode) -> list[dict]:
    starts, ends = candidate_boundaries(ep)
    result = []
    for a in starts:
        for b in ends:
            if b - a < 20: continue
            if b - a > 60: break
            item = score_candidate(ep, a, b)
            if item: result.append(item)
    return result


def overlap(a: dict, b: dict) -> float:
    s, t = a["segments"][0], b["segments"][0]
    if s["episode"] != t["episode"]: return 0
    lo = max(ts_to_s(s["start"]), ts_to_s(t["start"]))
    hi = min(ts_to_s(s["end"]), ts_to_s(t["end"]))
    if hi <= lo: return 0
    union = max(ts_to_s(s["end"]), ts_to_s(t["end"])) - min(ts_to_s(s["start"]), ts_to_s(t["start"]))
    return (hi - lo) / max(union, 0.001)


def select_promos(candidates: list[dict], episodes: list[Episode], count: int = 20,
                  max_per_episode: int = 1) -> list[dict]:
    if count <= 0: return []
    # Compare just the top ten per episode; keeps similarity-check cost bounded.
    by_episode = defaultdict(list)
    for c in candidates: by_episode[c["episode"]].append(c)
    pool = []
    for ep, group in by_episode.items():
        pool.extend(sorted(group, key=lambda c: -c["score"])[:max(10, max_per_episode * 5)])
    pool.sort(key=lambda c: (-c["score"], c["episode"], c["segments"][0]["start"]))
    by_num = {e.number: e for e in episodes}
    texts = []
    for c in pool:
        sg = c["segments"][0]; episode = by_num[c["episode"]]
        texts.append(" ".join(e.text for e in selected_dialogue(episode, ts_to_s(sg["start"]), ts_to_s(sg["end"]))))
    # TF-IDF is used only for near-duplicate filtering, not fixed genre keywords.
    similarities = None
    if len(texts) >= 2:
        try:
            matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=9000).fit_transform(texts)
            similarities = cosine_similarity(matrix)
        except ValueError:
            pass
    selected = []
    chosen_indices = []
    counts = Counter()
    for i, item in enumerate(pool):
        ep = item["episode"]
        if counts[ep] >= max_per_episode: continue
        if any(overlap(item, other) > .35 for other in selected): continue
        if similarities is not None and any(similarities[i, j] > .82 for j in chosen_indices): continue
        selected.append(dict(item)); chosen_indices.append(i); counts[ep] += 1
        if len(selected) >= count: break
    for rank, c in enumerate(selected, 1):
        c["rank"] = rank
        c["promo_id"] = f"P{rank:03d}"
    return selected


def flatten_promos(promos: Iterable[dict]) -> Iterable[dict]:
    for p in promos:
        seg = p["segments"][0]
        yield {
            "rank": p["rank"], "promo_id": p["promo_id"], "episode": p["episode"],
            "start": seg["start"], "end": seg["end"], "duration_seconds": p["duration_seconds"],
            "overall_score": p["score"], "hook_score": p["hook_score"],
            "cliffhanger_score": p["cliffhanger_score"], "self_contained_score": p["self_contained_score"],
            "scene_transitions": p["scene_transitions"], "hook": p["hook"],
            "cliffhanger": p["cliffhanger"], "why_it_works": p["why_it_works"],
            "quality_warnings": " | ".join(p["quality_warnings"]),
        }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows: return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_analysis(input_path: str | Path, output_dir: str | Path, count: int = 20,
                 max_per_episode: int = 1):
    if count < 1 or max_per_episode < 1:
        raise ValueError("Count and maximum promos per episode must be positive.")
    parsed = [parse_episode(name, txt, i) for i, (name, txt) in enumerate(read_inputs(input_path), 1)]
    parsed.sort(key=lambda e: e.number)
    dup = [n for n, ct in Counter(e.number for e in parsed).items() if ct > 1]
    if dup: raise ValueError(f"Duplicate episode numbers in uploaded files: {dup}")
    invalid = [e for e in parsed if len(selected_dialogue(e, 0, e.total_s)) < 3 or e.total_s < 20]
    if invalid:
        problems = ", ".join(f"episode {e.number} ({Path(e.name).name})" for e in invalid[:8])
        raise ValueError(f"Cannot parse enough timed script content in {problems}. Verify the timestamp format.")
    candidates = []
    coverage_counts = {}
    for ep in parsed:
        items = generate_candidates(ep)
        coverage_counts[ep.number] = len(items)
        candidates.extend(items)
    chosen = select_promos(candidates, parsed, count=count, max_per_episode=max_per_episode)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool_name": "Kuku Promo Finder v2",
        "run_mode": "Offline NLP + show-independent transparent scoring + TF-IDF deduplication",
        "scoring_note": "0-100 relative heuristic scores; NOT calibrated engagement or conversion predictions.",
        "clip_style": "Single continuous segments from one episode; stitched clips not supported.",
        "review_note": "Selections are unverified against footage and require editorial review.",
        "processed_episode_count": len(parsed),
        "total_candidate_count": len(candidates),
        "selected_promo_count": len(chosen),
        "requested_promo_count": count,
        "max_promos_per_episode": max_per_episode,
        "selected_promos": chosen,
    }
    (out / "promos_ranked.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out / "promos_ranked.csv", flatten_promos(chosen))
    write_csv(out / "episode_coverage.csv", [
        {"episode": e.number, "source_file": Path(e.name).name,
         "timed_events": len(e.events), "dialogue_lines": sum(x.kind == "DIALOGUE" for x in e.events),
         "scene_count": e.scene_count, "runtime_seconds": e.total_s,
         "candidates_generated": coverage_counts[e.number],
         "selected_promos": sum(p["episode"] == e.number for p in chosen),
         "parse_warnings": " | ".join(e.warnings)} for e in parsed
    ])
    grouped = defaultdict(list)
    for c in candidates: grouped[c["episode"]].append(c)
    write_csv(out / "candidate_scores_top3_per_episode.csv", [
        {"episode": ep.number, "candidate_rank": i, "start": c["segments"][0]["start"],
         "end": c["segments"][0]["end"], "duration_seconds": c["duration_seconds"],
         "score": c["score"], "hook": c["hook"], "cliffhanger": c["cliffhanger"],
         "quality_warnings": " | ".join(c["quality_warnings"])}
        for ep in parsed for i, c in enumerate(sorted(grouped[ep.number], key=lambda x: -x["score"])[:3], 1)
    ])
    write_csv(out / "editor_review_template.csv", [
        {"promo_id": p["promo_id"], "rank": p["rank"], "episode": p["episode"],
         "start": p["segments"][0]["start"], "end": p["segments"][0]["end"],
         "editor_decision": "", "hook_0_to_5": "", "cliffhanger_0_to_5": "",
         "standalone_0_to_5": "", "notes": ""} for p in chosen
    ])
    (out / "run_summary.json").write_text(json.dumps({
        "version": "2.0", "processed_episode_count": len(parsed),
        "total_candidate_count": len(candidates), "selected_promo_count": len(chosen),
        "chosen_with_scene_transitions": sum(p["scene_transitions"] > 0 for p in chosen),
        "chosen_with_warnings": sum(bool(p["quality_warnings"]) for p in chosen),
        "limitations": ["script-only", "heuristic scores", "human review required", "continuous clips only"],
    }, indent=2), encoding="utf-8")
    return payload, parsed, candidates


def main():
    p = argparse.ArgumentParser(description="Version 2: find ranked 20–60s promo suggestions in timestamped screenplays.")
    p.add_argument("input", help="ZIP, folder, or single TXT")
    p.add_argument("--out", default="outputs")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--per-episode", type=int, default=1)
    args = p.parse_args()
    payload, episodes, candidates = run_analysis(args.input, args.out, args.count, args.per_episode)
    print(f"Parsed {len(episodes)} episodes; generated {len(candidates)} valid candidate clips; selected {len(payload['selected_promos'])}")
    for item in payload["selected_promos"][:8]:
        sg = item["segments"][0]
        print(f"#{item['rank']:02d} ep{item['episode']:02d} {sg['start']}-{sg['end']} score={item['score']:.2f} warnings={len(item['quality_warnings'])}")
    print(f"Output saved in {args.out}")


if __name__ == "__main__": main()
