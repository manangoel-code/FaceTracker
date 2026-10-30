"""
Quick checks for the parts that are easy to break.
Run with:  py test_app.py   (it prints "All checks passed." at the end)
"""
import numpy as np

import app
from metrics import benchmark_summary

client = app.app.test_client()
LOCAL = "http://127.0.0.1:5000"


# Names become folder names, so anything that could leave the folder must be refused.
assert app.is_valid_name("Alex Smith")
assert not app.is_valid_name("../secret")
assert not app.is_valid_name("")
assert not app.is_valid_name("x" * 51)


# Benchmark numbers: one right frame and one wrong frame.
rows = [
    {"outcome": "correct_identification", "latency_ms": 10, "fps": 20},
    {"outcome": "wrong_identity", "latency_ms": 30, "fps": 10},
]
summary = benchmark_summary(rows)
assert summary["accuracy"] == 50.0
assert summary["precision"] == 50.0
assert summary["recall"] == 50.0
assert benchmark_summary([]) == {"samples": 0}


# Recognition with made-up face vectors (saved people are put back afterwards).
saved_people = dict(app.people)
app.people.clear()
alex = np.array([1, 0, 0], dtype=np.float32)
sam = np.array([0, 1, 0], dtype=np.float32)
app.people["Alex"] = [{"id": "a", "embedding": alex, "has_image": False}]
app.people["Sam"] = [{"id": "b", "embedding": sam, "has_image": False}]

assert app.recognize(alex)[0] == "Alex"                     # clear match
in_between = np.array([1, 1, 0], dtype=np.float32) / np.sqrt(2)
assert app.recognize(in_between)[0] is None                 # too close to call, so unknown

# Renaming someone onto a name that's taken must be refused, not crash.
response = client.patch("/people/Alex", json={"name": "Sam"}, base_url=LOCAL)
assert response.status_code == 409

app.people.clear()
app.people.update(saved_people)


# Other websites must not be able to use the app.
assert client.get("/status", base_url="http://evil.example").status_code == 403
response = client.post("/camera/stop", base_url=LOCAL, headers={"Origin": "http://evil.example"})
assert response.status_code == 403
assert client.get("/status", base_url=LOCAL).status_code == 200

print("All checks passed.")
