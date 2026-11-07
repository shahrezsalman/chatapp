# WhatsApp chat analyzer

Pure Python application using spaCy (sentence segmentation, linguistic analysis,
named entities and phrase matching) and dateparser (relative dates). Processing
is local after installation; no API key or hosted model is needed.

## Run

For the upload page, install the requirements and English model below, then run:

```powershell
python -m streamlit run app.py
```

Upload a UTF-8 `.txt` WhatsApp export, select its date format, and click
**Generate Report**. The page calls `whatsapp_analyzer.py`, displays the structured
report, and offers text/JSON downloads. Temporary upload files are cleaned up after
analysis; existing report files in the project are not overwritten.

Use Python 3.10 or newer. In a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m spacy download en_core_web_sm
.venv\Scripts\python whatsapp_analyzer.py example_chat.txt --project Apollo --output report.json
```

Export your WhatsApp group **without media**, then pass the exported `.txt` path
instead of `example_chat.txt`. Use `--date-order MDY` for month-first exports;
the default is day-first. Repeat `--project` for your known project names.

The JSON contains meetings, task assignments/requests, project updates, and
costs/quotations. Each candidate includes source text, sender, timestamp,
possible assignees, money expressions, entities, and interpreted date candidates.
One sentence can belong to multiple categories. Explicit urgency words set
priority; otherwise priority is unspecified. Cancellation, proposal, and past-event
labels help distinguish mentions from commitments.

## Deploy on Streamlit Community Cloud

Push `app.py`, `frontend.py`, `whatsapp_analyzer.py`, `structured_report.py`,
and `requirements.txt` to the same directory in your GitHub repository.
The requirements install the English spaCy model automatically during deployment.

In Streamlit Community Cloud, create an app, select your repository and branch,
and set **Main file path** to `app.py` (or its repository-relative path).
Select Python 3.13 in advanced settings to match the local development environment,
then deploy. No API keys or secrets are required.

Dependency setup reference:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies

## Extraction behavior and limits

The terminal displays **Meetings Scheduled** (details, date/time, location),
**Procurement Details** (product, quantity, cost), and **HR Details** (new
enrollment, salary issues, employee shifting). The same data is saved under
`summary` in the JSON; the original `events` remain available for tasks and project
updates. No new dependencies or online AI service are required.

Summary dates are quoted as written to avoid inventing missing times. Unknown
fields display `Not specified`. Meeting statuses distinguish proposals and
cancellations. HR entries preserve source messages, including proposed changes.
Procurement details retain price conditions and unit/total wording in `Details`.
Products are recognized using noun phrases, entities and an editable vocabulary
in `structured_report.py`. Prices without an explicit product may remain unassigned.
Multiple products in one sentence may share sentence-level prices; review allocation.
The summary does not reconcile references across messages into one final schedule.

This is an English NLP/rule-based baseline, not a trained business-event classifier.
It cannot guarantee completeness or importance. Roman Urdu and other languages
need language-specific rules/models. It does not inspect attachments, images,
voice notes or PDFs. Android/iOS timestamped text exports and multiline messages
are supported; unusual export formats may need a header-regex adjustment.

Review all candidates. Assignees and product entities can be missing or inaccurate;
currency symbols remain as written to avoid guessing a currency. Amounts may also
be budgets or payments, not product prices. Relative dates use the original message
timestamp; incomplete dates can have inferred components and no timezone is assumed.
Negation and tense handling are heuristic. Separate messages are not linked:
later changes do not automatically modify earlier meetings/tasks, and references
like “do that tomorrow” cannot reliably be resolved. Project names supplied on the
command line improve recognition of updates that omit the word “project”.

References: https://spacy.io/usage/models and https://dateparser.readthedocs.io/en/latest/
