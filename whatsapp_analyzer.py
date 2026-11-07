"""Local, English-language WhatsApp event extraction. No API keys required.

Usage: python whatsapp_analyzer.py chat.txt --date-order DMY --output report.json
Results are heuristic candidates, not verified commitments or a live calendar.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import dateparser
import spacy
from dateparser.search import search_dates
from spacy.matcher import PhraseMatcher
from structured_report import build_summary, render_summary


# Android: 06/11/2026, 09:30 - Alice: ...
# iOS: [06/11/2026, 09:30:00] Alice: ...
HEADER = re.compile(
    r"^\[?(?P<date>\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s*[APap][Mm])?)"
    r"(?:\]\s*|\s+-\s+)(?P<body>.*)$"
)
INVISIBLE = str.maketrans("", "", "\u200e\u200f\u202a\u202b\u202c\ufeff")
NUMBER = r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*[kKmM]\b)?"
CURRENCY = r"(?:PKR|INR|USD|EUR|GBP|AED|SAR|Rs\.?|[$€£₹])"
MONEY = re.compile(
    rf"(?<!\w)(?:{CURRENCY}\s*{NUMBER}|{NUMBER}\s*(?:{CURRENCY}|rupees|dollars|euros)\b)",
    re.I,
)
TIME_HINT = re.compile(
    r"\b(?:today|tomorrow|tonight|yesterday|next|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|january|february|march|april|may|june|july|august|"
    r"september|october|november|december|\d{1,2}\s*[ap]m|\d{1,2}:\d{2}|"
    r"\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}|in\s+\d+\s+(?:days?|hours?|weeks?))\b", re.I
)
ACTION_VERBS = {"send", "submit", "finish", "complete", "prepare", "review", "fix",
                "update", "deliver", "share", "create", "test", "deploy", "design",
                "implement", "check", "upload", "write", "contact", "build"}
TERMS = {
    "meeting": ["meeting", "meet", "sync", "standup", "stand-up", "demo", "conference", "call"],
    "task": ["assigned to", "responsible for", "take ownership", "your task", "action item"],
    "project": ["project", "release", "deployment", "sprint", "milestone", "build", "feature"],
    "update": ["completed", "finished", "deployed", "released", "blocked", "delayed",
               "in progress", "on track", "status", "update", "ready", "shipped", "approved"],
    "quotation": ["quote", "quotation", "price", "cost", "priced", "per unit", "unit price"],
}


@dataclass
class Message:
    timestamp: datetime
    sender: str
    text: str
    line: int


def parse_chat(text: str, date_order: str = "DMY") -> tuple[list[Message], list[str]]:
    """Preserve multiline messages and skip timestamped WhatsApp system notices."""
    messages, warnings = [], []
    current = None
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.translate(INVISIBLE).replace("\u202f", " ").replace("\xa0", " ")
        match = HEADER.match(line)
        if not match:
            if current is not None:
                current.text += "\n" + line
            elif line.strip():
                warnings.append(f"Unattached text at line {line_number} was skipped.")
            continue
        current = None
        stamp = dateparser.parse(
            f"{match['date']} {match['time']}", languages=["en"],
            settings={"DATE_ORDER": date_order, "PREFER_LOCALE_DATE_ORDER": False},
        )
        if stamp is None:
            warnings.append(f"Invalid timestamp at line {line_number}.")
            continue
        sender, separator, body = match["body"].partition(": ")
        if not separator:  # System event, not a participant message.
            continue
        current = Message(stamp, sender.strip(), body, line_number)
        messages.append(current)
    return messages, warnings


class ChatAnalyzer:
    def __init__(self, model: str = "en_core_web_sm", projects: list[str] | None = None,
                 date_order: str = "DMY"):
        self.nlp = spacy.load(model)
        self.date_order = date_order
        self.projects = projects or []
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for label, phrases in TERMS.items():
            self.matcher.add(label, list(self.nlp.pipe(phrases)))

    def dates(self, text: str, message: Message) -> list[dict]:
        if not TIME_HINT.search(text):
            return []
        # Mask amounts so a price like "USD 2027" is not read as a year.
        cleaned = MONEY.sub(lambda m: " " * len(m[0]), text)
        matches = search_dates(cleaned, languages=["en"], settings={
            "RELATIVE_BASE": message.timestamp,
            "DATE_ORDER": self.date_order,
            "PREFER_LOCALE_DATE_ORDER": False,
            "PREFER_DATES_FROM": "future",
        }) or []
        return [{"text": phrase, "interpreted_datetime": value.isoformat(),
                 "needs_review": True} for phrase, value in matches if TIME_HINT.search(phrase)]

    @staticmethod
    def status(text: str) -> str:
        if re.search(r"\b(cancel(?:led|ed)?|called off|no longer|won't|will not|not scheduled)\b", text, re.I):
            return "cancelled_or_negated"
        if re.search(r"\b(reschedul\w*|postpon\w*|moved to)\b", text, re.I):
            return "rescheduled"
        if "?" in text or re.search(r"\b(maybe|might|could|propose|tentative|shall we|let's)\b", text, re.I):
            return "proposed_or_requested"
        if re.search(r"\b(was|were|held|yesterday|last week|completed|finished)\b", text, re.I):
            return "past_or_completed"
        return "stated"

    def analyze(self, messages: list[Message]) -> list[dict]:
        people = sorted({m.sender for m in messages}, key=len, reverse=True)
        results = []
        for message, doc in zip(messages, self.nlp.pipe(m.text for m in messages)):
            for sentence in doc.sents:
                text = sentence.text.strip()
                if not text:
                    continue
                labels = {self.nlp.vocab.strings[key] for key, _, _ in self.matcher(sentence)}
                verbs = {t.lemma_.lower() for t in sentence if t.pos_ == "VERB"}
                named_projects = [p for p in self.projects if re.search(rf"(?<!\w){re.escape(p)}(?!\w)", text, re.I)]
                mentioned = [p for p in people if re.search(rf"(?<!\w)@?{re.escape(p)}(?!\w)", text, re.I)]
                request = bool(re.search(r"\b(please|kindly|must|need to|has to|have to|can you|could you|will you)\b", text, re.I))
                commitment = bool(re.search(r"\b(I will|I'll|I shall)\b", text, re.I))
                amounts = [m[0] for m in MONEY.finditer(text)]
                dates = self.dates(text, message)
                categories = []
                if "meeting" in labels:
                    categories.append("meeting")
                if "task" in labels or (verbs & ACTION_VERBS and (request or commitment or mentioned)):
                    categories.append("task")
                if ("project" in labels or named_projects) and "update" in labels:
                    categories.append("project_update")
                if "quotation" in labels or amounts:
                    categories.append("cost_or_quotation")
                if not categories:
                    continue
                state = self.status(text)
                entities = [{"text": e.text, "label": e.label_} for e in sentence.ents
                            if e.label_ in {"PERSON", "ORG", "PRODUCT", "MONEY", "GPE", "DATE", "TIME"}]
                # Dependency parsing supplies candidate owners; mentions alone are not proof.
                owners = []
                for token in sentence:
                    if token.lemma_.lower() in ACTION_VERBS:
                        owners.extend(child.text for child in token.children
                                      if child.dep_ in {"nsubj", "nsubjpass"} and child.pos_ == "PROPN")
                if commitment:
                    owners.append(message.sender)
                owners.extend(mentioned)
                if "task" in labels:
                    owner_match = re.search(r"(?:assigned to|responsible for)\s+(@?[\w]+)", text, re.I)
                    if owner_match and "assigned to" in owner_match[0].lower():
                        owners.append(owner_match[1])
                products = list(dict.fromkeys(e.text for e in sentence.ents if e.label_ == "PRODUCT"))
                results.append({
                    "categories": categories,
                    "status": state,
                    "priority": "high" if re.search(r"\b(urgent|important|mandatory|critical|asap)\b", text, re.I) else "unspecified",
                    "sender": message.sender,
                    "sent_at": message.timestamp.isoformat(),
                    "source_line": message.line,
                    "evidence": text,
                    "original_message": message.text,
                    "possible_assignees": list(dict.fromkeys(owners)) if "task" in categories else [],
                    "projects": named_projects,
                    "products": products,
                    "amounts": amounts,
                    "date_candidates": dates,
                    "entities": entities,
                    "needs_review": True,
                })
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chat", type=Path, help="WhatsApp .txt export (without media)")
    parser.add_argument("--output", type=Path, default=Path("report.json"))
    parser.add_argument("--date-order", choices=["DMY", "MDY", "YMD"], default="DMY")
    parser.add_argument("--project", action="append", default=[], help="Known project name; repeat as needed")
    parser.add_argument("--model", default="en_core_web_sm")
    args = parser.parse_args()
    if args.chat.resolve() == args.output.resolve():
        parser.error("Output must differ from the input chat.")
    if not args.chat.is_file():
        parser.error(
            f"Chat file not found: {args.chat.resolve()}\n"
            "Pass the path to your exported WhatsApp .txt file, or try example_chat.txt."
        )
    try:
        messages, warnings = parse_chat(args.chat.read_text(encoding="utf-8-sig"), args.date_order)
        if not messages:
            parser.error("No participant messages found. Check the export format and date order.")
        try:
            analyzer = ChatAnalyzer(args.model, args.project, args.date_order)
        except OSError as exc:
            parser.exit(1, f"Could not load NLP model {args.model!r}: {exc}\n"
                        f"Install the model with: python -m spacy download {args.model}\n")
        events = analyzer.analyze(messages)
        summary = build_summary(analyzer, messages, MONEY)
        counts = Counter(category for event in events for category in event["categories"])
        report = {"messages_analyzed": len(messages), "counts": dict(counts),
                  "warnings": warnings, "summary": summary, "events": events,
                  "notes": ["English heuristic extraction; review candidates against original messages.",
                            "Dates use the message timestamp as their reference; timezone is unspecified.",
                            "Missing date/time components may be inferred. These are not verified deadlines.",
                            "Events are message-level evidence, not a reconciled schedule. Later cancellations may supersede earlier events."]}
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        parser.exit(1, f"File error: {exc}\n")
    print(f"Analyzed {len(messages)} messages; extracted {len(events)} candidates.")
    print(render_summary(summary))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
