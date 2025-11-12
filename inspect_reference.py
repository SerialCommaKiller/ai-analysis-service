import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

import main
from run_tests import (
    connect_with_fallback,
    fetch_client_data,
    gather_pdf_candidates,
    resolve_query,
)


def load_environment() -> None:
    load_dotenv()


def pretty_print(title: str, data: Any) -> None:
    print(f"\n=== {title} ===")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2))
    else:
        print(data)


def main_cli(reference_id: str) -> None:
    load_environment()

    connection_string = os.getenv("DB_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError("DB_CONNECTION_STRING missing.")

    raw_query = os.getenv("DB_REFERENCE_QUERY")
    if not raw_query:
        raise RuntimeError("DB_REFERENCE_QUERY missing.")
    query = resolve_query(raw_query)

    base_dir = Path(__file__).with_name("agent_files")
    pdf_paths = gather_pdf_candidates(base_dir, reference_id)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found for reference {reference_id}")

    with connect_with_fallback(connection_string) as connection:
        cursor = connection.cursor()
        client_data = fetch_client_data(cursor, query, reference_id)

    pretty_print("Client Data", client_data)

    for pdf_path in pdf_paths:
        print(f"\n>>> Processing {pdf_path.name}")
        pdf_bytes = pdf_path.read_bytes()
        payload, validation, recording_info = main.run_pipeline(
            pdf_bytes,
            "Recorded Mortgage",
            client_data,
            return_details=True,
        )

        pretty_print("Recording Info", recording_info)
        pretty_print("Validation", validation)
        pretty_print("Payload", payload)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python inspect_reference.py <reference_id>")
    main_cli(sys.argv[1])
