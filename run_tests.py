import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import pyodbc
from dotenv import load_dotenv

import main


def load_env() -> None:
    load_dotenv()


def resolve_query(raw_query: str) -> str:
    query = (raw_query or "").strip()
    if query.startswith('"') and query.endswith('"'):
        query = query[1:-1].strip()
    return query


def fetch_client_data(cursor: pyodbc.Cursor, query: str, reference_id: str) -> Dict[str, Any]:
    cursor.execute(query, reference_id)
    row = cursor.fetchone()
    if not row:
        return {}
    columns = [column[0] for column in cursor.description]
    data = {columns[index]: row[index] for index in range(len(columns))}
    for key, value in list(data.items()):
        if isinstance(value, datetime):
            data[key] = value.strftime("%m/%d/%Y")
        elif value is not None and not isinstance(value, str):
            data[key] = str(value)
    # Normalize expected keys for pipeline
    payload: Dict[str, Any] = {
        "reference_id": reference_id,
        "borrower": data.get("borrower"),
        "borrower_name": data.get("borrower"),
        "property_address": data.get("property_address"),
        "loan_amount": data.get("loan_amount"),
        "note_date": data.get("note_date"),
        "mers_min": data.get("min"),
        "instruction": data.get("instruction"),
    }
    return {key: value for key, value in payload.items() if value not in (None, "")}


def gather_pdf_candidates(base_dir: Path, reference_id: str) -> List[Path]:
    pattern = f"*{reference_id}*.pdf"
    return sorted(path for path in base_dir.glob(pattern) if path.is_file())


def connect_with_fallback(conn_str: str) -> pyodbc.Connection:
    try:
        return pyodbc.connect(conn_str)
    except pyodbc.InterfaceError as exc:
        if "ODBC Driver 18 for SQL Server" in conn_str and "IM002" in str(exc):
            fallback = conn_str.replace("ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server")
            return pyodbc.connect(fallback)
        raise


def main_cli() -> None:
    load_env()

    connection_string = os.getenv("DB_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError("DB_CONNECTION_STRING missing.")

    raw_query = os.getenv("DB_REFERENCE_QUERY")
    if not raw_query:
        raise RuntimeError("DB_REFERENCE_QUERY missing.")
    query = resolve_query(raw_query)

    references = [
        "1346087606",
        "1652815719",
        "40158",
        "171675712",
        "8310013031",
    ]

    base_dir = Path(__file__).with_name("agent_files")
    results: Dict[str, Any] = {}

    with connect_with_fallback(connection_string) as connection:
        cursor = connection.cursor()
        for reference_id in references:
            pdf_paths = gather_pdf_candidates(base_dir, reference_id)
            client_data = fetch_client_data(cursor, query, reference_id)

            if not pdf_paths:
                results[reference_id] = {"error": "no pdf found"}
                continue

            payload = main.select_pdf_with_stamp(pdf_paths, "Recorded Mortgage", client_data)
            results[reference_id] = payload

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main_cli()
