"""Run with: python -m streamlit run frontend.py"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from structured_report import render_summary


def generate_report(filename: str, content: bytes, date_order: str = "DMY") -> dict:
    """Run the existing analyzer with isolated temporary input/output files."""
    if Path(filename).suffix.lower() != ".txt":
        raise ValueError("Please upload a .txt file.")
    if not content.strip():
        raise ValueError("The uploaded file is empty.")
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("Please upload a file smaller than 10 MB.")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Please save the chat as a UTF-8 text file and upload it again.") from exc
    if "\x00" in decoded:
        raise ValueError("This file does not appear to be a UTF-8 text chat.")
    script = Path(__file__).with_name("whatsapp_analyzer.py")
    with tempfile.TemporaryDirectory(prefix="whatsapp_report_") as directory:
        chat = Path(directory) / "chat.txt"
        output = Path(directory) / "report.json"
        chat.write_text(decoded, encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONIOENCODING"] = "utf-8"
        try:
            result = subprocess.run(
                [sys.executable, str(script), str(chat), "--date-order", date_order,
                 "--output", str(output)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                env=environment, timeout=300,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError("Analysis timed out. Please try a smaller chat export.") from exc
        if result.returncode:
            raise ValueError(result.stderr.strip() or "The analyzer could not generate a report.")
        return json.loads(output.read_text(encoding="utf-8"))


def main():
    import streamlit as st

    st.set_page_config(page_title="Whatsapp Chat Analyzer Application", page_icon="💬", layout="centered")
    st.title("Whatsapp Chat Analyzer Application")
    st.write("Upload your exported WhatsApp chat to generate meeting, procurement, and HR details.")

    def clear_report():
        st.session_state.pop("chat_report", None)

    uploaded = st.file_uploader("Upload WhatsApp chat (.txt)", type=["txt"],
                                accept_multiple_files=False, on_change=clear_report,
                                help="Export the chat without media. UTF-8 text, up to 10 MB.")
    date_order = st.selectbox("Chat date format", ["DMY", "MDY", "YMD"],
                             format_func=lambda v: {"DMY": "Day / Month / Year", "MDY": "Month / Day / Year",
                                                    "YMD": "Year / Month / Day"}[v],
                             on_change=clear_report)
    if st.button("Generate Report", type="primary", disabled=uploaded is None):
        clear_report()
        try:
            with st.spinner("Analyzing your chat…"):
                st.session_state["chat_report"] = generate_report(uploaded.name, uploaded.getvalue(), date_order)
        except (ValueError, OSError) as exc:
            st.error(str(exc))

    report = st.session_state.get("chat_report")
    if report is not None:
        st.success(f"Report generated from {report['messages_analyzed']} messages.")
        for warning in report.get("warnings", []):
            st.warning(warning)
        summary_text = render_summary(report["summary"])
        st.subheader("Your report")
        st.text(summary_text)
        st.download_button("Download report (.txt)", summary_text, "whatsapp_report.txt", "text/plain")
        st.download_button("Download report (.json)", json.dumps(report, indent=2, ensure_ascii=False),
                           "whatsapp_report.json", "application/json")


if __name__ == "__main__":
    main()
