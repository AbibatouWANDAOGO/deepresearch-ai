"""Report export: JSON (the full record), Markdown and PDF.

    python -m app.reports.export --research-id <id>      # re-render a saved report

Text that comes from the web (claims, quotes, source titles) is escaped in both formats, so
a page cannot inject links, images or HTML into the report. The PDF is laid out by PyMuPDF
from HTML, with its embedded Unicode fonts (no extra dependency).
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from app.agents.fact_checker import Status
from app.agents.writer import CitedStatement, Report, render_statement
from app.reports.i18n import localize

LABELS = {
    "fr": {
        "question": "Question", "research": "Recherche", "generated": "générée le", "stopped": "arrêt",
        "summary": "Résumé", "unverified": "Points non vérifiés", "limitations": "Limites",
        "evidence": "Traçabilité des affirmations", "sources": "Sources", "published": "publié le",
        "collected": "consulté le", "supports": "soutient", "contradicts": "contredit",
        "why": "Justification", "counts": "Affirmations",
        Status.VERIFIED: "Vérifié", Status.PARTIALLY_VERIFIED: "Partiellement vérifié",
        Status.CONFLICTING: "Contesté", Status.UNVERIFIED: "Non vérifié",
    },
    "en": {
        "question": "Question", "research": "Research", "generated": "generated on", "stopped": "stopped",
        "summary": "Summary", "unverified": "Unverified points", "limitations": "Limitations",
        "evidence": "Claim traceability", "sources": "Sources", "published": "published",
        "collected": "accessed", "supports": "supports", "contradicts": "contradicts",
        "why": "Justification", "counts": "Claims",
        Status.VERIFIED: "Verified", Status.PARTIALLY_VERIFIED: "Partially verified",
        Status.CONFLICTING: "Disputed", Status.UNVERIFIED: "Unverified",
    },
}

# Escaping brackets is enough to break links and images, and "<" to break HTML and autolinks.
_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>|~])")
_OWN_CITATION = re.compile(r"\\\[(S\d+)\\\]")

def labels(report: Report) -> dict:
    return LABELS.get(report.language, LABELS["en"])


def md_escape(text: str) -> str:
    """Neutralize Markdown syntax in untrusted text (links, images, HTML, headings, tables)."""
    text = _MD_SPECIAL.sub(r"\\\1", " ".join(text.split()))
    return re.sub(r"^([#>+\-=]|\d+[.)])", r"\\\1", text)  # no heading, quote or list at line start


def sentence_join(parts: list[str]) -> str:
    """'Title ?' + 'site' -> 'Title ? site.': a period only where the part has no end mark."""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out:
            out += " " if out[-1] in ".!?" else ". "
        out += part
    return out + ("" if out.endswith((".", "!", "?")) else ".")


def md_cited(text: str) -> str:
    """Escape text that already carries our [S#] citations, keeping those readable."""
    return _OWN_CITATION.sub(r"[\1]", md_escape(text))


def safe_url(url: str) -> str | None:
    return url if re.match(r"^https?://[^\s<>\"]+$", url) else None


def _counts(report: Report, t: dict) -> str:
    colon = " :" if report.language == "fr" else ":"
    return ", ".join(f"{t[Status(s)].lower()}{colon} {n}" for s, n in report.status_counts.items() if n)


def _all_statements(report: Report) -> list[CitedStatement]:
    return [s for section in report.sections for s in section.statements] + report.uncertainties


# --- Markdown ----------------------------------------------------------------------------


def to_markdown(report: Report) -> str:
    t = labels(report)
    lines = [f"# {md_escape(report.title)}", "",
             f"*{t['question']} : {md_escape(report.question)}*" if report.language == "fr"
             else f"*{t['question']}: {md_escape(report.question)}*"]
    meta = [f"{t['research']} {report.research_id}" if report.research_id else t["research"],
            f"{t['generated']} {report.generated_at.date().isoformat()}"]
    if report.stop_reason:
        reason = localize(report.stop_reason, report.language)
        meta.append(f"{t['stopped']} : {reason}" if report.language == "fr" else f"{t['stopped']}: {reason}")
    if _counts(report, t):
        meta.append(f"{t['counts']} : {_counts(report, t)}" if report.language == "fr"
                    else f"{t['counts']}: {_counts(report, t)}")
    lines += [f"*{' · '.join(meta)}*", "", f"## {t['summary']}", "", md_cited(report.executive_summary)]

    for section in report.sections:
        lines += ["", f"## {md_escape(section.heading)}", ""]
        if section.statements:
            lines += [f"- {md_cited(render_statement(s, report.language))}" for s in section.statements]
        else:
            lines.append(md_escape(section.body))

    if report.uncertainties:
        lines += ["", f"## {t['unverified']}", ""]
        lines += [f"- {md_escape(s.text)}"
                  + (f" — *{md_escape(localize(s.justification, report.language))}*" if s.justification else "")
                  for s in report.uncertainties]
    if report.limitations:
        lines += ["", f"## {t['limitations']}", ""] + [f"- {md_escape(x)}" for x in report.limitations]

    evidence = [s for s in _all_statements(report) if s.citations]
    if evidence:
        lines += ["", f"## {t['evidence']}", ""]
        for s in evidence:
            lines.append(f"**{s.claim_id} · {t[s.status]}** — {md_escape(s.text)}")
            lines.append("")
            for c in s.citations:
                where = f", {c.location}" if c.location else ""
                quote = f" : « {md_escape(c.quote)} »" if c.quote and report.language == "fr" else (
                    f': "{md_escape(c.quote)}"' if c.quote else "")
                lines.append(f"- [{c.label}]{where} ({t[c.role]}){quote}")
            if s.justification:
                why = md_cited(s.justification)
                lines.append(f"- *{t['why']} : {why}*" if report.language == "fr" else f"- *{t['why']}: {why}*")
            lines.append("")
        lines.pop()

    if report.bibliography:
        lines += ["", f"## {t['sources']}", ""]
        for b in report.bibliography:
            parts = [md_escape(b.title), md_escape(b.site_name or "")]
            if b.published_date:
                parts.append(f"{t['published']} {md_escape(b.published_date)}")
            parts.append(f"{t['collected']} {b.collected_at}")
            url = safe_url(b.url)
            lines.append(f"- [{b.label}] {sentence_join(parts)}" + (f" <{url}>" if url else ""))
    return "\n".join(lines) + "\n"


# --- HTML and PDF ------------------------------------------------------------------------

CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.4; color: #1f2328; }
h1 { font-size: 18pt; margin-bottom: 4pt; }
h2 { font-size: 13pt; margin-top: 14pt; color: #0b3d63; }
p.meta { font-size: 8.5pt; color: #57606a; }
li { margin-bottom: 3pt; }
p.claim { margin-top: 8pt; margin-bottom: 2pt; }
ul.evidence { font-size: 8.5pt; color: #3b434b; }
span.cite { color: #0b5cad; }
"""


def _h(text: str) -> str:
    return html.escape(" ".join(text.split()))


def _h_cited(text: str) -> str:
    return re.sub(r"\[(S\d+)\]", r'<span class="cite">[\1]</span>', _h(text))


def to_html(report: Report) -> str:
    t = labels(report)
    colon = " :" if report.language == "fr" else ":"
    meta = [f"{t['research']} {_h(report.research_id)}" if report.research_id else t["research"],
            f"{t['generated']} {report.generated_at.date().isoformat()}"]
    if report.stop_reason:
        meta.append(f"{t['stopped']}{colon} {_h(localize(report.stop_reason, report.language))}")
    if _counts(report, t):
        meta.append(f"{t['counts']}{colon} {_h(_counts(report, t))}")
    out = [f"<h1>{_h(report.title)}</h1>",
           f"<p class='meta'><i>{t['question']}{colon} {_h(report.question)}</i><br/>{' · '.join(meta)}</p>",
           f"<h2>{t['summary']}</h2>", f"<p>{_h_cited(report.executive_summary)}</p>"]
    for section in report.sections:
        out.append(f"<h2>{_h(section.heading)}</h2>")
        if section.statements:
            out.append("<ul>" + "".join(f"<li>{_h_cited(render_statement(s, report.language))}</li>"
                                        for s in section.statements) + "</ul>")
        else:
            out.append(f"<p>{_h(section.body)}</p>")
    if report.uncertainties:
        out.append(f"<h2>{t['unverified']}</h2><ul>" + "".join(
            f"<li>{_h(s.text)}" + (f" — <i>{_h(localize(s.justification, report.language))}</i>"
                                    if s.justification else "") + "</li>"
            for s in report.uncertainties) + "</ul>")
    if report.limitations:
        out.append(f"<h2>{t['limitations']}</h2><ul>" + "".join(f"<li>{_h(x)}</li>" for x in report.limitations)
                   + "</ul>")
    evidence = [s for s in _all_statements(report) if s.citations]
    if evidence:
        out.append(f"<h2>{t['evidence']}</h2>")
        for s in evidence:
            out.append(f"<p class='claim'><b>{s.claim_id} · {t[s.status]}</b> — {_h(s.text)}</p><ul class='evidence'>")
            for c in s.citations:
                where = f", {_h(c.location)}" if c.location else ""
                quote = f"{colon} « {_h(c.quote)} »" if c.quote else ""
                out.append(f"<li>[{c.label}]{where} ({t[c.role]}){quote}</li>")
            if s.justification:
                out.append(f"<li><i>{t['why']}{colon} {_h_cited(s.justification)}</i></li>")
            out.append("</ul>")
    if report.bibliography:
        out.append(f"<h2>{t['sources']}</h2><ul>")
        for b in report.bibliography:
            parts = [_h(b.title), _h(b.site_name or "")]
            if b.published_date:
                parts.append(f"{t['published']} {_h(b.published_date)}")
            parts.append(f"{t['collected']} {b.collected_at}")
            url = safe_url(b.url)
            out.append(f"<li>[{b.label}] {sentence_join(parts)}" + (f"<br/>{_h(url)}" if url else "") + "</li>")
        out.append("</ul>")
    return f"<html><head><style>{CSS}</style></head><body>{''.join(out)}</body></html>"


def to_pdf(report: Report, path: Path) -> Path:
    import pymupdf

    story = pymupdf.Story(html=to_html(report))
    page = pymupdf.paper_rect("a4")
    frame = page + (56, 56, -56, -56)
    writer = pymupdf.DocumentWriter(str(path))
    more = True
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(frame)
        story.draw(device)
        writer.end_page()
    writer.close()
    _number_pages(path)
    return path


def _number_pages(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open(str(path))
    for i, page in enumerate(doc, start=1):
        page.insert_text((page.rect.width - 90, page.rect.height - 28), f"{i} / {doc.page_count}",
                         fontsize=8, color=(0.4, 0.4, 0.4))
    doc.saveIncr()
    doc.close()


# --- Saving ------------------------------------------------------------------------------


def save_report(report: Report, directory: Path) -> tuple[dict[str, Path], str | None]:
    """Write report.json, report.md and report.pdf. A PDF failure is returned, not raised:
    the JSON and Markdown are what matters."""
    directory.mkdir(parents=True, exist_ok=True)
    files = {"json": directory / "report.json", "md": directory / "report.md"}
    files["json"].write_text(report.model_dump_json(indent=2), encoding="utf-8")
    files["md"].write_text(to_markdown(report), encoding="utf-8")
    try:
        files["pdf"] = to_pdf(report, directory / "report.pdf")
    except Exception as exc:  # noqa: BLE001 - a layout problem must not lose the report
        return files, f"PDF export failed: {exc}"
    return files, None


def load_report(path: Path) -> Report:
    return Report.model_validate(json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="Re-render a saved report as Markdown and PDF.")
    parser.add_argument("--research-id", required=True)
    args = parser.parse_args(argv)
    directory = get_settings().reports_dir / args.research_id
    source = directory / "report.json"
    if not source.exists():
        parser.error(f"no saved report at {source}")
    files, error = save_report(load_report(source), directory)
    for kind, path in files.items():
        print(f"{kind}: {path}")
    if error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
