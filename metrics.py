"""
Benchmark calculations: turns the recorded rows into the numbers on the Benchmark page.

Every row is one webcam frame, with an "outcome" that is one of:
  correct_identification  the expected person was recognized
  correct_rejection       nobody was expected and no face was found
  wrong_identity          a face was given the wrong name
  missed_detection        someone was expected but no face was found
  unknown_face            a face was found but not recognized
  false_detection         nobody was expected but a face was found
"""

OUTCOMES = [
    "correct_identification",
    "correct_rejection",
    "wrong_identity",
    "missed_detection",
    "unknown_face",
    "false_detection",
]


def percent(part, whole):
    """part / whole as a percentage, or None if whole is 0."""
    if whole == 0:
        return None
    return round(100 * part / whole, 1)


def benchmark_summary(rows):
    total = len(rows)
    if total == 0:
        return {"samples": 0}

    # Count how many rows had each outcome.
    counts = {}
    for outcome in OUTCOMES:
        counts[outcome] = sum(1 for row in rows if row["outcome"] == outcome)

    correct = counts["correct_identification"]
    rejected = counts["correct_rejection"]
    false_matches = counts["wrong_identity"] + counts["false_detection"]
    misses = counts["missed_detection"] + counts["unknown_face"] + counts["wrong_identity"]

    latencies = sorted(row["latency_ms"] for row in rows)
    speeds = [row["fps"] for row in rows]
    p95_index = min(total - 1, int(0.95 * total))

    return {
        "samples": total,
        "accuracy": percent(correct + rejected, total),
        "precision": percent(correct, correct + false_matches),  # None = nothing to measure
        "recall": percent(correct, correct + misses),
        "false_matches": false_matches,
        "missed_faces": counts["missed_detection"],
        "unknown_faces": counts["unknown_face"],
        "avg_fps": round(sum(speeds) / total, 1),
        "min_fps": min(speeds),
        "max_fps": max(speeds),
        "avg_latency_ms": round(sum(latencies) / total, 1),
        "p95_latency_ms": latencies[p95_index],
        "outcomes": counts,
    }
