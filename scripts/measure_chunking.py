"""Measure M1 on each original document; no API calls or answer generation."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.m1_chunking import (chunk_basic, chunk_hierarchical, chunk_semantic,
                              chunk_structure_aware, load_documents)


def main():
    documents = load_documents()
    report = {"num_documents": len(documents), "measurement": "per_document",
              "size_unit": "characters", "semantic_threshold": 0.85,
              "strategies": {}}
    for name, chunker in [("basic", chunk_basic), ("semantic", chunk_semantic),
                          ("hierarchical", chunk_hierarchical),
                          ("structure", chunk_structure_aware)]:
        started = time.perf_counter()
        chunks, parent_count = [], 0
        for document in documents:
            result = chunker(document["text"], metadata=document["metadata"])
            if name == "hierarchical":
                parents, result = result
                parent_count += len(parents)
            chunks.extend(result)
        lengths = [len(chunk.text) for chunk in chunks]
        stats = {"count": len(chunks), "avg_length": round(sum(lengths) / len(lengths), 2)
                 if lengths else 0, "max_length": max(lengths, default=0),
                 "seconds": round(time.perf_counter() - started, 3)}
        if name == "hierarchical":
            stats["parent_count"] = parent_count
        report["strategies"][name] = stats
        print(name, stats, flush=True)
    report["timing_note"] = "Semantic timing includes its first model load in this process."
    destination = ROOT / "reports" / "m1_chunking_report.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {destination}")


if __name__ == "__main__":
    main()
