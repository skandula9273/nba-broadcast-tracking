"""Numbers-consistency checker: make README/artifact drift impossible to ship.

The README states quantitative claims; `eval_results/*.json` hold the truth; the two drift apart silently
between runs. This tool maps each README number to its source artifact + JSON path via a manifest
(`docs/numbers_manifest.yaml`) and fails loudly on any mismatch — so a PR that edits a README number without
its artifact, or commits a new artifact without updating the README, breaks CI instead of a reviewer's eyes.

It catches four drift modes and reports ALL of them (never just the first):
  1. MISMATCH      — the artifact no longer produces the number the README shows (within tolerance).
  2. MISSING ANCHOR — the README no longer contains the exact claim text (someone edited the number/prose).
  3. STALE FAMILY  — a newer run of the same artifact family exists than the one the README cites, and the
                     citation is NOT marked deliberate (`pin_reason`). This is "committed a new artifact,
                     forgot to update the README".
  4. UNSOURCED     — a quantitative token in the README maps to no manifest entry at all (and isn't an
                     acknowledged non-artifact number). Listed, never auto-deleted.

Honest older-run citations are first-class: a claim with `pin_reason` deliberately cites a specific (often
older) run, which is expressible and checkable rather than indistinguishable from staleness.

Dependency-light on purpose (stdlib + pyyaml, a core dep) so it drops into any repo and runs in CI.
Run:  python -m hoopvec.eval.check_numbers        (or `make check-numbers`)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TIMESTAMP_RE = re.compile(r"_\d{8}T\d{6}Z\.json$")   # eval_results filename stamp: <family>_YYYYMMDDThhmmssZ.json
# Number-like tokens in the README that the audit accounts for. Deliberately broad; `ignore_patterns` +
# `exempt` in the manifest carve out the non-metric ones (years, versions, structural counts, ...).
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


@dataclass
class Issue:
    kind: str          # MISMATCH | MISSING ANCHOR | STALE FAMILY | UNSOURCED | BAD PATH | MANIFEST
    claim_id: str
    detail: str


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)     # non-failing notes (deliberate pins, exemptions)
    checked: int = 0

    def fail(self, kind: str, claim_id: str, detail: str) -> None:
        self.issues.append(Issue(kind, claim_id, detail))

    @property
    def ok(self) -> bool:
        return not self.issues


# ----------------------------------------------------------------------------------------------------------
# JSON-path resolution.  Slash-separated (NOT dot — artifact keys like "coverage@0.25" contain dots).
# A segment is a dict key, or an integer index into a list.  e.g. "results/sweeps/id_swap/4/trained/recall@1"
# ----------------------------------------------------------------------------------------------------------
def resolve(obj, path: str):
    cur = obj
    for seg in path.split("/"):
        if isinstance(cur, list):
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            if seg not in cur:
                raise KeyError(f"key {seg!r} not found (have: {sorted(cur)[:8]})")
            cur = cur[seg]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at {seg!r}")
    return cur


def read_value(artifact_json: dict, claim: dict):
    """Return the numeric value a claim points at: either a JSON path or a derived op over two paths."""
    if "derived" in claim:
        d = claim["derived"]
        op = d["op"]
        if op == "divide":
            return float(resolve(artifact_json, d["of"])) / float(resolve(artifact_json, d["by"]))
        if op == "subtract":
            return float(resolve(artifact_json, d["of"])) - float(resolve(artifact_json, d["by"]))
        raise ValueError(f"unknown derived op {op!r}")
    return float(resolve(artifact_json, claim["path"]))


def family_of(filename: str) -> str | None:
    """`serving_latency_20260802T175803Z.json` -> `serving_latency`, or None if it carries no timestamp."""
    return TIMESTAMP_RE.sub("", filename) if TIMESTAMP_RE.search(filename) else None


def latest_in_family(artifacts_dir: Path, filename: str) -> str | None:
    """Newest committed file sharing this file's family (exact prefix + a timestamp), by lexical stamp order."""
    fam = family_of(filename)
    if fam is None:
        return None
    exact = re.compile(rf"^{re.escape(fam)}_\d{{8}}T\d{{6}}Z\.json$")
    peers = sorted(p.name for p in artifacts_dir.glob(f"{fam}_*.json") if exact.match(p.name))
    return peers[-1] if peers else None


# ----------------------------------------------------------------------------------------------------------
# Core checks
# ----------------------------------------------------------------------------------------------------------
def run_command(cmd: str) -> float:
    """Run a manifest-declared command (trusted, from the committed manifest) and return the last number it
    prints. For counts the eval_results manifest can't hold — e.g. `git rev-list --count HEAD`, test counts."""
    import subprocess
    out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)  # noqa: S602 (trusted)
    nums = re.findall(r"-?\d+(?:\.\d+)?", out.stdout)
    if not nums:
        raise ValueError(f"no number in output of {cmd!r}: {out.stdout!r} {out.stderr[:120]!r}")
    return float(nums[-1])


def compare_value(actual: float, claim: dict, cid: str, source: str, rep: Report) -> None:
    display, tol = float(claim["display"]), float(claim["tol"])
    if abs(actual - display) > tol:
        rep.fail("MISMATCH", cid,
                 f"README shows {display} but {source} = {round(actual, 6)} "
                 f"(|Δ|={round(abs(actual - display), 6)} > tol {tol})")


def check_claim(claim: dict, readme: str, artifacts_dir: Path, rep: Report) -> None:
    cid = claim.get("id", "<no id>")
    rep.checked += 1

    # (2) anchors: the exact claim text must still be in the README.
    for anchor in claim.get("anchors", []):
        if anchor not in readme:
            rep.fail("MISSING ANCHOR", cid, f"README no longer contains {anchor!r} — was the number edited?")

    # Command-sourced claim (no artifact, no stale-family check): e.g. the test count from `grep def test_`.
    if "command" in claim:
        try:
            actual = run_command(claim["command"]) * float(claim.get("scale", 1))
        except (ValueError, OSError) as e:
            rep.fail("BAD PATH", cid, f"command failed: {e}")
            return
        compare_value(actual, claim, cid, f"`{claim['command']}`", rep)
        return

    artifact_name = claim["artifact"]
    artifact_path = artifacts_dir / artifact_name
    if not artifact_path.exists():
        rep.fail("BAD PATH", cid, f"artifact {artifact_name} not found in {artifacts_dir}/")
        return
    data = json.loads(artifact_path.read_text())

    # (1) mismatch: the artifact must still produce the displayed value (within tolerance, after scale).
    try:
        actual = read_value(data, claim) * float(claim.get("scale", 1))
    except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
        rep.fail("BAD PATH", cid, f"cannot read value from {artifact_name}: {e}")
        return
    src = claim.get("derived", claim.get("path"))
    compare_value(actual, claim, cid, f"{artifact_name}:{src}", rep)

    # (3) stale family: is a newer run of this family committed than the one cited?
    newest = latest_in_family(artifacts_dir, artifact_name)
    if newest and newest != artifact_name:
        if "pin_reason" in claim:
            rep.infos.append(f"  · {cid}: deliberately pins {artifact_name} (newer {newest} exists) — "
                             f"{claim['pin_reason'].strip().splitlines()[0]}")
        else:
            rep.fail("STALE FAMILY", cid,
                     f"cites {artifact_name} but a newer run {newest} exists — update the README to cite it, "
                     f"or add pin_reason: to keep the old one deliberately")
    elif "pin_reason" in claim:
        # Deliberate pin, but the pinned file IS the latest -> the reason is now moot; surface it quietly.
        rep.infos.append(f"  · {cid}: pin_reason set but {artifact_name} is already the latest in its family")


_FENCE = re.compile(r"```.*?```", re.DOTALL)   # fenced code / mermaid / demo transcript
_INLINE = re.compile(r"`[^`\n]+`")             # inline code spans (paths, config names)
_URL = re.compile(r"\]\([^)]*\)")              # markdown link / image / badge targets


def mask_noise(text: str) -> str:
    """Blank out non-prose regions (same length, so offsets still align with the original) so the numeric
    audit sees only prose + tables — where reader-facing CLAIMS live. Fenced blocks (bash/mermaid/the demo
    transcript), inline code, and link/badge URLs are commands/diagrams/output, NOT quantitative claims."""
    def blank(m: re.Match) -> str:
        return " " * (m.end() - m.start())
    for pat in (_FENCE, _INLINE, _URL):
        text = pat.sub(blank, text)
    return text


def audit_unsourced(readme: str, manifest: dict, rep: Report) -> None:
    """Every README prose/table number must be covered by a claim/exempt anchor, or match an ignore pattern.
    Else it's listed as UNSOURCED (never auto-deleted). Fenced/code/URL regions are excluded (see mask_noise)."""
    covered: list[tuple[int, int]] = []
    for entry in manifest.get("claims", []) + manifest.get("exempt", []):
        for anchor in entry.get("anchors", []):
            start = readme.find(anchor)
            while start != -1:
                covered.append((start, start + len(anchor)))
                start = readme.find(anchor, start + 1)

    ignore = [re.compile(p) for p in manifest.get("ignore_patterns", [])]
    masked = mask_noise(readme)   # tokenize the masked copy; positions match the original (equal length)

    def is_covered(pos: int) -> bool:
        return any(a <= pos < b for a, b in covered)

    seen: set[str] = set()
    for m in NUMBER_RE.finditer(masked):
        if is_covered(m.start()):
            continue
        line = readme[readme.rfind("\n", 0, m.start()) + 1: readme.find("\n", m.end())]
        ctx = readme[max(0, m.start() - 14): m.end() + 8]   # from the original, for readable ctx + ignore match
        if any(p.search(ctx) for p in ignore):
            continue
        key = f"{m.group()}  |  {ctx.strip()}"
        if key not in seen:
            seen.add(key)
            rep.fail("UNSOURCED", f"line: {line.strip()[:80]}", f"'{m.group()}' has no manifest mapping")


def run(readme_path: Path, manifest_path: Path, artifacts_dir: Path) -> Report:
    rep = Report()
    manifest = yaml.safe_load(manifest_path.read_text())
    readme = readme_path.read_text()

    seen_ids: set[str] = set()
    for claim in manifest.get("claims", []):
        cid = claim.get("id", "<no id>")
        if cid in seen_ids:
            rep.fail("MANIFEST", cid, "duplicate claim id")
        seen_ids.add(cid)
        check_claim(claim, readme, artifacts_dir, rep)

    for entry in manifest.get("exempt", []):
        for anchor in entry.get("anchors", []):
            if anchor not in readme:
                rep.infos.append(f"  · exempt anchor gone (ok if intentional): {anchor!r}")
        rep.infos.append(f"  · exempt: {entry.get('anchors', ['?'])[0]!r} — {entry.get('reason', '')}")

    audit_unsourced(readme, manifest, rep)
    return rep


def render(rep: Report, verbose: bool) -> str:
    out: list[str] = []
    by_kind: dict[str, list[Issue]] = {}
    for i in rep.issues:
        by_kind.setdefault(i.kind, []).append(i)

    for kind in ("MISMATCH", "MISSING ANCHOR", "STALE FAMILY", "BAD PATH", "UNSOURCED", "MANIFEST"):
        items = by_kind.get(kind, [])
        if not items:
            continue
        out.append(f"\n✗ {kind} ({len(items)})")
        for i in items:
            out.append(f"    [{i.claim_id}] {i.detail}")

    if verbose and rep.infos:
        out.append("\nℹ notes")
        out.extend(rep.infos)

    if rep.ok:
        out.append(f"\n✓ all {rep.checked} claims consistent with eval_results/  "
                   f"({len(rep.infos)} notes; run -v to see deliberate pins + exemptions)")
    else:
        out.append(f"\n✗ {len(rep.issues)} issue(s) across {rep.checked} claims — "
                   f"README and eval_results/ disagree (or a number is unsourced). Fix above.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Check README numbers against eval_results/ artifacts")
    ap.add_argument("--readme", default="README.md", type=Path)
    ap.add_argument("--manifest", default="docs/numbers_manifest.yaml", type=Path)
    ap.add_argument("--artifacts", default="eval_results", type=Path, help="dir of committed *.json artifacts")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print deliberate pins + exemptions")
    args = ap.parse_args()

    for p in (args.readme, args.manifest, args.artifacts):
        if not p.exists():
            print(f"error: {p} not found (run from the repo root, or pass --{p})", file=sys.stderr)
            sys.exit(2)

    rep = run(args.readme, args.manifest, args.artifacts)
    print(render(rep, args.verbose))
    sys.exit(0 if rep.ok else 1)


if __name__ == "__main__":
    main()
