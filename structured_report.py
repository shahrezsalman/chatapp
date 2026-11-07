"""Extractive business summaries using the existing local spaCy pipeline."""
import re

from spacy.matcher import PhraseMatcher


TERMS = {
    "meeting": ["meeting", "meetings", "appointment", "appointments", "standup",
                "demo", "conference", "compensation review", "internal review",
                "recruitment readiness review"],
    "enrollment": ["recruitment", "new joiners", "new hire", "new-hire", "onboarding",
                   "enrollment", "enrolment", "enrolments", "enrollments", "internees",
                   "interns", "junior developers", "joining date", "induction", "hired"],
    "salary": ["salary", "salaries", "payroll", "compensation", "increment", "wages",
               "unpaid", "pay raise", "pay cut"],
    "shifting": ["transferred", "transfer", "relocated", "relocation", "reassigned",
                 "shift change", "night shift", "day shift", "employee shifting"],
}
PRODUCTS = {"table", "laptop", "chair", "printer", "monitor", "computer", "phone",
            "desk", "server", "router", "equipment", "furniture", "toner", "paper",
            "vehicle", "keyboard", "mouse", "software", "license", "licence"}


def build_summary(analyzer, messages, money_pattern):
    matcher = PhraseMatcher(analyzer.nlp.vocab, attr="LOWER")
    for key, phrases in TERMS.items():
        matcher.add(key, [analyzer.nlp.make_doc(p) for p in phrases])

    def labels(span):
        return {analyzer.nlp.vocab.strings[k] for k, _, _ in matcher(span)}

    summary = {"meetings_scheduled": [], "procurement_details": [],
               "hr_details": {"new_enrollment": [], "salary_issues": [], "shifting_of_employees": []}}
    for message, doc in zip(messages, analyzer.nlp.pipe(m.text for m in messages)):
        source = {"sender": message.sender, "sent_at": message.timestamp.isoformat(),
                  "source_line": message.line, "needs_review": True}
        message_labels = labels(doc)
        # Keep whole HR messages: names, conditions and approvals often follow a colon
        # or appear in the next sentence. Do not invent a person from the sender.
        for label, field in [("enrollment", "new_enrollment"), ("salary", "salary_issues"),
                             ("shifting", "shifting_of_employees")]:
            if label in message_labels:
                if label == "shifting" and not any(t.lemma_.lower() in {
                    "employee", "staff", "worker", "team", "department", "shift", "office"
                } for t in doc) and not any(e.label_ == "PERSON" for e in doc.ents):
                    continue
                summary["hr_details"][field].append({"details": message.text, **source})

        last_product = None  # Only carry explicit context within this message.
        for sentence in doc.sents:
            sentence_labels = labels(sentence)
            text = sentence.text.strip()
            # Do not interpret "meeting expectations" or preparing meeting briefs
            # as scheduled meetings.
            meeting = "meeting" in sentence_labels and not re.search(
                r"meeting expectations|meeting briefs?|meeting updates?|meeting schedule|meeting room", text, re.I)
            if "meeting room" in text.lower() and re.search(r"appointment|confirmed|scheduled", text, re.I):
                meeting = True
            if meeting:
                # Semicolon-separated schedules retain their explicitly stated context.
                parts = text.split(";")
                for part in parts:
                    part_doc = analyzer.nlp(part.strip())
                    date_text = [e.text for e in part_doc.ents if e.label_ in {"DATE", "TIME"}]
                    if not date_text:
                        date_text = [d["text"] for d in analyzer.dates(part, message)]
                    locations = []
                    for token in part_doc:
                        if token.lower_ not in {"at", "in", "on", "via"}:
                            continue
                        for obj in token.children:
                            if obj.dep_ != "pobj":
                                continue
                            subtree = list(obj.subtree)
                            candidate = part_doc[min(t.i for t in subtree):max(t.i for t in subtree) + 1]
                            if candidate.end < len(part_doc) and part_doc[candidate.end].text.rstrip(".").isalnum() and (
                                candidate[-1].lemma_.lower() in {"room", "hall", "building"}
                            ):
                                candidate = part_doc[candidate.start:candidate.end + 1]
                            if any(t.ent_type_ in {"DATE", "TIME"} for t in candidate):
                                continue
                            if any(t.lemma_.lower() in {"room", "office", "hall", "headquarters", "secretariat",
                                                       "zoom", "teams", "meet", "venue", "building"}
                                   or t.ent_type_ in {"GPE", "FAC", "LOC"} for t in candidate):
                                locations.append(candidate.text.rstrip("."))
                    state = analyzer.status(text)
                    if re.search(r"\b(tentative|proposed|pending)\b", text, re.I):
                        state = "proposed_or_requested"
                    summary["meetings_scheduled"].append({
                        "meeting_details": part.strip(), "date_and_time": "; ".join(date_text) or None,
                        "location": "; ".join(dict.fromkeys(locations)) or None,
                        "status": state, **source})

            amounts = [m[0] for m in money_pattern.finditer(text)]
            product_chunks = []
            for chunk in sentence.noun_chunks:
                if chunk.root.lemma_.lower() in PRODUCTS or (
                    chunk.root.dep_ in {"dobj", "obj"} and
                    chunk.root.head.lemma_.lower() in {"buy", "purchase", "procure", "order"}
                ):
                    if chunk.root.head.lower_ == "per":
                        continue
                    # Remove quantity/determiners while preserving product modifiers.
                    product_tokens = [t for t in chunk if t.dep_ not in {"det", "nummod", "poss"}]
                    product = " ".join(t.text for t in product_tokens).strip()
                    quantity = " ".join(t.text for t in chunk if t.dep_ == "nummod") or None
                    if product:
                        product_chunks.append((product, quantity))
            product_chunks.extend((e.text, None) for e in sentence.ents if e.label_ == "PRODUCT"
                                  and not money_pattern.search(e.text)
                                  and not any(e.text in p for p, _ in product_chunks))
            product_chunks = list(dict.fromkeys(product_chunks))
            procurement_context = amounts or re.search(
                r"\b(procurement|quotation|quote|cost|price|budget|purchase|buy|order|required quantity|reserve)\b", text, re.I)
            # HR money is compensation, not a product quotation.
            if procurement_context and ("salary" not in sentence_labels or product_chunks):
                if not product_chunks and amounts:
                    # A combined estimate must not inherit a single product.
                    explicit_products = list(dict.fromkeys(t.text for t in sentence
                                             if t.lemma_.lower() in PRODUCTS))
                    product_chunks = [(" + ".join(explicit_products) or last_product, None)]
                for product, quantity in product_chunks:
                    summary["procurement_details"].append({
                        "product": product, "quantity": quantity,
                        "cost": "; ".join(amounts) or None,
                        "details": text,
                        "cost_scope": "sentence amounts; review allocation" if len(product_chunks) > 1 else "see details for unit/total and conditions",
                        **source})
            if len(product_chunks) == 1 and product_chunks[0][0]:
                last_product = product_chunks[0][0]
            elif len(product_chunks) > 1:
                last_product = None
    return summary


def render_summary(summary):
    def value(v):
        return v or "Not specified"

    lines = ["1. Meetings Scheduled:"]
    if not summary["meetings_scheduled"]:
        lines.append("   No meeting details found.")
    for meeting in summary["meetings_scheduled"]:
        lines.extend([f"   Meeting details: {meeting['meeting_details']}",
                      f"   Date and Time: {value(meeting['date_and_time'])}",
                      f"   Location: {value(meeting['location'])}",
                      f"   Status: {meeting['status'].replace('_', ' ')}", ""])
    lines.append("2. Procurement Details:")
    if not summary["procurement_details"]:
        lines.append("   No procurement details found.")
    for item in summary["procurement_details"]:
        lines.extend([f"   Product: {value(item['product'])}",
                      f"   Quantity: {value(item['quantity'])}",
                      f"   Cost: {value(item['cost'])}",
                      f"   Details: {item['details']}", ""])
    lines.append("3. HR Details:")
    for label, key in [("New Enrollment", "new_enrollment"), ("Salary Issues", "salary_issues"),
                       ("Shifting of employees", "shifting_of_employees")]:
        lines.append(f"   {label}:")
        rows = summary["hr_details"][key]
        lines.extend([f"     - {row['details']}" for row in rows] or ["     Not found in the chat."])
    lines.extend(["", "Review extracted details. Dates/times are quoted as written; later messages may supersede earlier ones."])
    return "\n".join(lines)
