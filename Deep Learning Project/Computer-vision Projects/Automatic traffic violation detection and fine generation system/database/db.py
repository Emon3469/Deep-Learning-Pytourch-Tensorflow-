from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from services.pydentic import FineRecord


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = OUTPUT_DIR / "traffic_fines.db"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                track_id INTEGER,
                plate_number TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                fine_amount INTEGER NOT NULL,
                evidence_path TEXT NOT NULL,
                bbox TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_fine(record: FineRecord, db_path: Path = DB_PATH) -> int:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO fines (
                timestamp,
                frame_index,
                track_id,
                plate_number,
                violation_type,
                fine_amount,
                evidence_path,
                bbox
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.timestamp,
                record.frame_index,
                record.track_id,
                record.plate_number,
                record.violation_type,
                record.fine_amount,
                record.evidence_path,
                json.dumps(record.bbox),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def save_fines(records: list[FineRecord], db_path: Path = DB_PATH) -> list[int]:
    return [save_fine(record, db_path) for record in records]


def load_fines(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM fines ORDER BY id DESC").fetchall()

    fines = [dict(row) for row in rows]
    for fine in fines:
        try:
            fine["bbox"] = json.loads(fine["bbox"])
        except (TypeError, json.JSONDecodeError):
            fine["bbox"] = []
    return fines
