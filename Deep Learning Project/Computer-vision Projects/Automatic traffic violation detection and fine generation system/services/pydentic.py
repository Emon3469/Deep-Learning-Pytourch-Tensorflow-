from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.xyxy),
            "track_id": self.track_id,
        }


@dataclass(slots=True)
class FineRecord:
    timestamp: str
    frame_index: int
    track_id: int | None
    plate_number: str
    violation_type: str
    fine_amount: int
    evidence_path: str
    bbox: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "plate_number": self.plate_number,
            "violation_type": self.violation_type,
            "fine_amount": self.fine_amount,
            "evidence_path": self.evidence_path,
            "bbox": list(self.bbox),
        }
