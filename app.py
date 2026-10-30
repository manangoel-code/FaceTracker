"""
FaceTracker - recognize saved faces through your webcam.

Run it with:   py app.py
Then open:     http://127.0.0.1:5000

This file is split into parts:
  Part 1 - Saved people
  Part 2 - Recognizing faces
  Part 3 - The webcam
  Part 4 - Main page, status and settings
  Part 5 - Managing saved people
  Part 6 - Benchmark tests
  Part 7 - Importing photos
"""
import base64
import csv
import io
import json
import re
import shutil
import threading
import time
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from metrics import benchmark_summary
from vision import (face_quality, find_faces, group_similar_faces, load_models,
                    measure_sharpness, normalize, similarity)

APP_FOLDER = Path(__file__).resolve().parent
SAMPLES_FOLDER = APP_FOLDER / "saved_samples"
SAMPLES_FOLDER.mkdir(exist_ok=True)
SETTINGS_FILE = APP_FOLDER / "settings.json"

PORT = 5000
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 120 * 1024 * 1024  # biggest upload allowed (120 MB)


@app.before_request
def only_this_computer():
    """Stop other websites from using FaceTracker through your browser."""
    # A wrong address means a trick ("DNS rebinding") to make another site look like this one.
    if request.host not in ALLOWED_HOSTS:
        abort(403)
    # Browsers say which site sent a request. Refuse changes sent by any other site.
    origin = request.headers.get("Origin")
    if request.method != "GET" and origin and origin.split("//")[-1] not in ALLOWED_HOSTS:
        abort(403)


@app.errorhandler(413)
def upload_too_big(error):
    return jsonify(error="Those photos are too big together (120 MB max). Try fewer at once."), 413

# The webcam runs in its own thread. The lock stops two parts of the
# program from changing the same data at the same moment.
lock = threading.Lock()

# Recognition settings. These can be changed on the Live Camera page.
settings = {
    "threshold": 0.47,      # how similar a face must be to count as a match
    "margin": 0.07,         # how far the best match must be ahead of the second best
    "detection_every": 1,   # look for faces on every Nth frame
    "width": 640,           # frames are shrunk to this width before looking for faces
}

# Saved people, loaded from the saved_samples folder:
# {"Alex": [{"id": "1a2b3c4d5e6f", "embedding": face_vector, "has_image": True}, ...]}
people = {}

# Everything about the webcam that the page needs to know.
camera = {
    "on": False,          # True while the user wants the camera running
    "thread": None,
    "running": False,     # True once the camera has actually started
    "error": "",
    "faces": [],          # faces found in the latest frame
    "jpeg": None,         # latest frame with boxes drawn on it
    "fps": 0,
    "latency_ms": 0,
    "frame_count": 0,
}

# Faces we follow from frame to frame, so each one keeps the same number.
tracks = {}
next_track_id = 1

# Benchmark recording.
MAX_BENCHMARK_ROWS = 10000
benchmark = {
    "active": False,
    "rows": [],
    "expected": None,     # the person who should be in view (None = nobody)
    "started": 0,
    "lighting": "Normal lighting",
    "movement": "Still",
    "distance": "Normal",
    "notes": "",
}

# Faces found in imported photos, waiting for the user to check and save them.
photo_previews = {}


# ===============================================================
# Part 1 - Saved people
# ===============================================================
# Every person has a folder inside saved_samples.
# Every sample is two files with the same random id:
#   1a2b3c4d5e6f.npz  the face vector
#   1a2b3c4d5e6f.jpg  the face picture

def is_valid_name(name):
    """Names may only use letters, numbers, spaces, - and _ (up to 50 characters)."""
    return isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9 _-]{1,50}", name.strip()) is not None


def person_folder(name):
    return SAMPLES_FOLDER / name.strip()


def find_existing_name(name):
    """If "alex" is typed but "Alex" is already saved, use "Alex".
    (Windows folders ignore upper/lower case, so they would clash.)"""
    for saved_name in people:
        if saved_name.lower() == name.lower():
            return saved_name
    return name


def load_people():
    """Read every saved face vector from the saved_samples folder."""
    people.clear()
    for folder in sorted(SAMPLES_FOLDER.iterdir()):
        if not folder.is_dir() or not is_valid_name(folder.name):
            continue

        samples = []
        for file in sorted(folder.glob("*.npz")):
            try:
                with np.load(file) as data:
                    embedding = data["embedding"].astype(np.float32).flatten()
            except Exception as error:
                print("Skipping a broken sample:", file, error)
                continue

            if embedding.size < 32 or not np.isfinite(embedding).all():
                continue
            samples.append({
                "id": file.stem,
                "embedding": normalize(embedding),
                "has_image": file.with_suffix(".jpg").exists(),
            })
        people[folder.name] = samples


def save_sample(name, embedding, face_image):
    """Save one face (vector + picture) into that person's folder."""
    sample_id = uuid.uuid4().hex[:12]
    folder = person_folder(name)
    folder.mkdir(exist_ok=True)

    np.savez_compressed(folder / (sample_id + ".npz"), embedding=embedding)
    has_image = face_image is not None and face_image.size > 0
    if has_image:
        cv2.imwrite(str(folder / (sample_id + ".jpg")), face_image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    people.setdefault(name, []).append({"id": sample_id, "embedding": embedding, "has_image": has_image})
    return sample_id


load_people()


# ===============================================================
# Part 2 - Recognizing faces
# ===============================================================

def recognize(embedding):
    """Compare a face with every saved person.

    Returns (name, closest_name, score). name is None if we're not sure enough.
    """
    scores = []
    for name, samples in people.items():
        if samples:
            best = max(similarity(embedding, sample["embedding"]) for sample in samples)
            scores.append((best, name))

    if not scores:
        return None, None, 0.0

    scores.sort(reverse=True)
    best_score, best_name = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else -1

    # A match needs a high enough score AND must clearly beat the next person.
    high_enough = best_score >= settings["threshold"]
    clear_winner = best_score - second_score >= settings["margin"]
    if high_enough and clear_winner:
        return best_name, best_name, best_score
    return None, best_name, best_score


def get_track_id(x, y, size, now, taken_ids):
    """Give a face the number of the nearest face from a moment ago, or a new number."""
    global next_track_id
    best_id = None
    best_distance = float("inf")

    for track_id, track in tracks.items():
        distance = np.hypot(x - track["x"], y - track["y"])
        seen_recently = now - track["seen"] < 1.5
        close_enough = distance < max(size, track["size"]) * 0.85
        if track_id not in taken_ids and seen_recently and close_enough and distance < best_distance:
            best_id = track_id
            best_distance = distance

    if best_id is None:
        best_id = next_track_id
        next_track_id += 1

    tracks[best_id] = {"x": x, "y": y, "size": size, "seen": now}
    return best_id


def analyze_frame(frame, detector, recognizer, now):
    """Find, number and recognize every face in one webcam frame."""
    faces = []
    taken_ids = set()

    for face in find_faces(frame, detector, recognizer):
        x, y, w, h = face["box"]
        track_id = get_track_id(x + w / 2, y + h / 2, max(w, h), now, taken_ids)
        taken_ids.add(track_id)

        name, closest, score = recognize(face["embedding"])
        face["id"] = track_id
        face["name"] = name
        face["closest"] = closest
        face["similarity"] = round(score, 4)
        faces.append(face)

    # Forget faces we haven't seen for 2 seconds.
    for track_id in list(tracks):
        if now - tracks[track_id]["seen"] > 2:
            del tracks[track_id]
    return faces


def draw_boxes(frame, faces):
    """Draw a box and a name tag around every face."""
    green = (94, 168, 64)    # OpenCV colours are (blue, green, red)
    amber = (40, 150, 235)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for face in faces:
        x, y, w, h = face["box"]
        if face["name"]:
            color = green
            label = f"{face['name']}  {max(0, round(face['similarity'] * 100))}% similarity"
        else:
            color = amber
            label = f"Face {face['id']}  |  unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Name tag above the box
        (text_width, _), _ = cv2.getTextSize(label, font, 0.48, 1)
        tag_bottom = max(23, y)
        tag_right = min(frame.shape[1] - 1, x + text_width + 12)
        cv2.rectangle(frame, (x, tag_bottom - 23), (tag_right, tag_bottom), color, -1)
        cv2.putText(frame, label, (x + 6, tag_bottom - 7), font, 0.48, white, 1, cv2.LINE_AA)
    return frame


# ===============================================================
# Part 3 - The webcam
# ===============================================================

def open_webcam():
    """Open the first webcam. On Windows, DirectShow usually starts faster."""
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not webcam.isOpened():
        webcam = cv2.VideoCapture(0)
    return webcam


def camera_loop():
    """Runs in the background: read a frame, find faces, draw boxes, repeat."""
    webcam = None
    try:
        try:
            detector, recognizer = load_models()
        except Exception as error:
            camera["error"] = str(error)
            return

        webcam = open_webcam()
        if not webcam.isOpened():
            camera["error"] = "The camera could not be opened. Close other apps that use it."
            return

        webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        camera["running"] = True

        previous_time = time.monotonic()
        failed_reads = 0
        while camera["on"]:
            success, frame = webcam.read()
            if not success:
                # About 2 seconds of failures means the camera was unplugged or taken by another app.
                failed_reads += 1
                if failed_reads > 100:
                    camera["error"] = "Lost the camera. Is it still plugged in?"
                    break
                time.sleep(0.02)
                continue
            failed_reads = 0
            start_time = time.monotonic()

            # Mirror the picture, then shrink it if it's wider than the chosen width.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            target_width = settings["width"]
            if width > target_width:
                frame = cv2.resize(frame, (target_width, round(height * target_width / width)))

            with lock:
                # Looking for faces is the slow part, so we can skip some frames.
                # On skipped frames we keep showing the boxes from last time.
                analyzed = camera["frame_count"] % settings["detection_every"] == 0
                if analyzed:
                    camera["faces"] = analyze_frame(frame, detector, recognizer, start_time)
                faces = camera["faces"]

                camera["frame_count"] += 1
                camera["fps"] = round(1 / max(start_time - previous_time, 0.001), 1)
                camera["latency_ms"] = round((time.monotonic() - start_time) * 1000, 1)
                previous_time = start_time

                if analyzed:
                    record_benchmark_row(faces)

                picture = draw_boxes(frame, faces)
                ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    camera["jpeg"] = jpeg.tobytes()
    finally:
        # This runs however the loop ends, so the camera can always be started again.
        if webcam is not None:
            webcam.release()
        with lock:
            camera["running"] = False
            camera["thread"] = None
            camera["faces"] = []
            camera["jpeg"] = None


def video_stream():
    """Send webcam pictures to the browser one after another (an MJPEG stream)."""
    last_sent = None
    while camera["on"] and not camera["error"]:
        picture = camera["jpeg"]
        if picture is not None and picture is not last_sent:
            last_sent = picture
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + picture + b"\r\n"
        time.sleep(0.03)


@app.post("/camera/start")
def start_camera():
    with lock:
        camera["on"] = True
        camera["error"] = ""
        if camera["thread"] is None:
            camera["thread"] = threading.Thread(target=camera_loop, daemon=True)
            camera["thread"].start()
    return jsonify(ok=True)


@app.post("/camera/stop")
def stop_camera():
    camera["on"] = False
    return jsonify(ok=True)


@app.get("/video")
def video():
    return Response(video_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ===============================================================
# Part 4 - Main page, status and settings
# ===============================================================

@app.get("/")
def home():
    return render_template("index.html")


@app.get("/status")
def status():
    """The page asks for this every second to keep everything up to date."""
    with lock:
        faces = []
        for face in camera["faces"]:
            faces.append({
                "id": face["id"],
                "name": face["name"],
                "closest": face["closest"],
                "similarity": face["similarity"],
                "box": face["box"],
            })

        saved_people = []
        for name in sorted(people):
            saved_people.append({"name": name, "samples": len(people[name])})

        return jsonify(
            running=camera["running"],
            error=camera["error"],
            fps=camera["fps"],
            latency_ms=camera["latency_ms"],
            frames=camera["frame_count"],
            faces=faces,
            people=saved_people,
            settings=settings,
            benchmark={
                "active": benchmark["active"],
                "samples": len(benchmark["rows"]),
                "expected": benchmark["expected"],
            },
        )


def apply_settings(data):
    """Check new settings and use them. Returns an error message, or None if all is fine."""
    try:
        threshold = float(data.get("threshold", settings["threshold"]))
        margin = float(data.get("margin", settings["margin"]))
        every = int(data.get("detection_every", settings["detection_every"]))
        width = int(data.get("width", settings["width"]))
    except (ValueError, TypeError):
        return "Settings must be numbers."

    if not 0.30 <= threshold <= 0.85:
        return "Match threshold must be between 0.30 and 0.85."
    if not 0 <= margin <= 0.25:
        return "Runner-up margin must be between 0 and 0.25."
    if every not in (1, 2, 3) or width not in (480, 640, 960):
        return "Please pick one of the listed options."

    settings["threshold"] = threshold
    settings["margin"] = margin
    settings["detection_every"] = every
    settings["width"] = width
    return None


def load_settings():
    """Use the settings saved last time, if there are any."""
    if not SETTINGS_FILE.exists():
        return
    try:
        error = apply_settings(json.loads(SETTINGS_FILE.read_text()))
    except (ValueError, AttributeError):
        error = "the file is not valid"
    if error:
        print("Ignoring settings.json:", error)


load_settings()


@app.post("/settings")
def update_settings():
    data = request.get_json(silent=True) or {}
    with lock:
        error = apply_settings(data)
        if error:
            return jsonify(error=error), 400
        # Save to a file so the settings are kept next time the app starts.
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
    return jsonify(ok=True, settings=settings)


# ===============================================================
# Part 5 - Managing saved people
# ===============================================================

@app.post("/enroll")
def enroll():
    """Save the chosen face from the webcam as a sample for a person."""
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not is_valid_name(name):
        return jsonify(error="Enter a name using letters, numbers, spaces, - or _ (up to 50)."), 400

    try:
        face_id = int(data.get("face_id", -1))
    except (ValueError, TypeError):
        face_id = -1

    with lock:
        faces = camera["faces"]
        chosen = None
        for face in faces:
            if face["id"] == face_id:
                chosen = face
        if chosen is None and len(faces) == 1:
            chosen = faces[0]  # only one face in view, so it must be that one
        if chosen is None:
            return jsonify(error="Choose which face to save first."), 400

        name = find_existing_name(name)
        save_sample(name, chosen["embedding"], chosen["crop"])
        return jsonify(ok=True, name=name, samples=len(people[name]))


@app.get("/people/<name>/samples")
def list_samples(name):
    with lock:
        if name not in people:
            return jsonify(error="Person not found."), 404
        samples = [{"id": s["id"], "has_image": s["has_image"]} for s in people[name]]
        return jsonify(name=name, samples=samples)


@app.get("/people/<name>/samples/<sample_id>/image")
def sample_image(name, sample_id):
    if name not in people or not re.fullmatch(r"[0-9a-f]{12}", sample_id):
        abort(404)
    path = person_folder(name) / (sample_id + ".jpg")
    if not path.is_file():
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@app.delete("/people/<name>/samples/<sample_id>")
def delete_sample(name, sample_id):
    with lock:
        if name not in people or sample_id not in [s["id"] for s in people[name]]:
            return jsonify(error="Sample not found."), 404
        folder = person_folder(name)
        (folder / (sample_id + ".npz")).unlink(missing_ok=True)
        (folder / (sample_id + ".jpg")).unlink(missing_ok=True)
        people[name] = [s for s in people[name] if s["id"] != sample_id]
    return jsonify(ok=True)


@app.delete("/people/<name>")
def delete_person(name):
    with lock:
        if name not in people:
            return jsonify(error="Person not found."), 404
        shutil.rmtree(person_folder(name))
        del people[name]
    return jsonify(ok=True)


@app.patch("/people/<name>")
def rename_person(name):
    data = request.get_json(silent=True) or {}
    new_name = str(data.get("name", "")).strip()
    if not is_valid_name(new_name):
        return jsonify(error="Enter a name using letters, numbers, spaces, - or _ (up to 50)."), 400

    with lock:
        if name not in people:
            return jsonify(error="Person not found."), 404
        if new_name == name:
            return jsonify(ok=True)
        # Only a change of upper/lower case may reuse a name; anything else would overwrite someone.
        if new_name.lower() != name.lower() and find_existing_name(new_name) in people:
            return jsonify(error="That name is already used."), 409
        person_folder(name).rename(person_folder(new_name))
        people[new_name] = people.pop(name)
    return jsonify(ok=True)


# ===============================================================
# Part 6 - Benchmark tests
# ===============================================================
# While a test is running, every analyzed webcam frame becomes one row.
# The user tells us who *should* be in view; we record what was seen.

CSV_COLUMNS = [
    "timestamp", "elapsed_s", "expected", "predicted", "correct", "outcome",
    "detected_faces", "face_id", "similarity", "fps", "latency_ms",
    "lighting", "movement", "distance", "notes",
    "threshold", "margin", "width", "detection_every",
]


def record_benchmark_row(faces):
    """Add one row for this frame (only while a test is running)."""
    if not benchmark["active"]:
        return

    expected = benchmark["expected"]
    first_face = faces[0] if faces else None

    # What did we see? A name, an unknown face, or no face at all.
    if first_face is None:
        seen = "No face"
    elif first_face["name"] is None:
        seen = "Unknown"
    else:
        seen = first_face["name"]

    if expected is None:
        correct = seen == "No face"
        outcome = "correct_rejection" if correct else "false_detection"
    else:
        correct = seen == expected
        if correct:
            outcome = "correct_identification"
        elif seen == "No face":
            outcome = "missed_detection"
        elif seen == "Unknown":
            outcome = "unknown_face"
        else:
            outcome = "wrong_identity"

    best_similarity = max([face["similarity"] for face in faces], default=None)
    now = time.time()
    benchmark["rows"].append({
        "timestamp": round(now, 3),
        "elapsed_s": round(now - benchmark["started"], 2),
        "expected": expected or "None",
        "predicted": seen,
        "correct": int(correct),
        "outcome": outcome,
        "detected_faces": len(faces),
        "face_id": first_face["id"] if first_face else "",
        "similarity": round(best_similarity, 4) if best_similarity is not None else "",
        "fps": camera["fps"],
        "latency_ms": camera["latency_ms"],
        "lighting": benchmark["lighting"],
        "movement": benchmark["movement"],
        "distance": benchmark["distance"],
        "notes": benchmark["notes"],
        "threshold": settings["threshold"],
        "margin": settings["margin"],
        "width": settings["width"],
        "detection_every": settings["detection_every"],
    })

    if len(benchmark["rows"]) >= MAX_BENCHMARK_ROWS:
        benchmark["active"] = False


def read_test_conditions(data):
    """Copy the lighting/movement/distance labels from the page into the benchmark."""
    benchmark["lighting"] = str(data.get("lighting", benchmark["lighting"]))[:60]
    benchmark["movement"] = str(data.get("movement", benchmark["movement"]))[:60]
    benchmark["distance"] = str(data.get("distance", benchmark["distance"]))[:60]


@app.post("/benchmark/start")
def start_benchmark():
    data = request.get_json(silent=True) or {}
    expected = data.get("expected")
    with lock:
        if expected != "None" and expected not in people:
            return jsonify(error="Choose a saved person, or nobody."), 400
        benchmark["active"] = True
        benchmark["rows"] = []
        benchmark["expected"] = None if expected == "None" else expected
        benchmark["started"] = time.time()
        benchmark["notes"] = str(data.get("notes", ""))[:160]
        read_test_conditions(data)
    return jsonify(ok=True)


@app.post("/benchmark/mark")
def update_benchmark():
    """Change who is expected (or the conditions) while a test is running."""
    data = request.get_json(silent=True) or {}
    expected = data.get("expected")
    with lock:
        if not benchmark["active"]:
            return jsonify(error="Start a test first."), 400
        if expected != "None" and expected not in people:
            return jsonify(error="Choose a saved person, or nobody."), 400
        benchmark["expected"] = None if expected == "None" else expected
        read_test_conditions(data)
    return jsonify(ok=True)


@app.post("/benchmark/stop")
def stop_benchmark():
    with lock:
        benchmark["active"] = False
    return jsonify(ok=True)


@app.get("/benchmark/summary")
def get_benchmark_summary():
    with lock:
        return jsonify(benchmark_summary(benchmark["rows"]))


@app.get("/benchmark/export")
def export_benchmark():
    with lock:
        rows = list(benchmark["rows"])

    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        text.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": 'attachment; filename="facetracker_benchmark.csv"'},
    )


# ===============================================================
# Part 7 - Importing photos
# ===============================================================
# Step 1: the user uploads photos and we suggest groups of matching faces.
# Step 2: the user checks and names the groups, then we save them.

MAX_PHOTOS = 30
MAX_PHOTO_BYTES = 12 * 1024 * 1024  # 12 MB
PHOTO_TYPES = (".jpg", ".jpeg", ".png", ".webp")


def read_uploaded_photo(upload):
    """Turn an uploaded file into an image OpenCV can use (at most 1600 pixels wide/tall)."""
    if Path(upload.filename or "").suffix.lower() not in PHOTO_TYPES:
        raise ValueError("Only JPG, PNG and WEBP photos can be used.")

    data = upload.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise ValueError("Each photo must be smaller than 12 MB.")

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{upload.filename} could not be opened.")

    height, width = image.shape[:2]
    if max(height, width) > 1600:
        scale = 1600 / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    return image


def to_data_url(image):
    """Turn an image into text the browser can show directly in an <img>."""
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


@app.post("/photo-import/preview")
def preview_photos():
    if request.form.get("permission") != "yes":
        return jsonify(error="Please confirm you have permission to use these photos."), 400

    uploads = request.files.getlist("photos")
    if not 1 <= len(uploads) <= MAX_PHOTOS:
        return jsonify(error=f"Choose between 1 and {MAX_PHOTOS} photos."), 400

    # Find every usable face in every photo.
    try:
        detector, recognizer = load_models()
        faces = []
        for photo_number, upload in enumerate(uploads, start=1):
            image = read_uploaded_photo(upload)
            for face in find_faces(image, detector, recognizer, min_size=25):
                face["sharpness"] = round(measure_sharpness(face["crop"]), 1)
                if face["sharpness"] < 12:
                    continue  # too blurry to be useful
                face["id"] = uuid.uuid4().hex[:12]
                face["photo"] = photo_number
                faces.append(face)
    except (ValueError, RuntimeError) as error:
        return jsonify(error=str(error)), 400

    # Group faces that look alike. Keep at most 120 faces, 20 per group.
    best_faces = sorted(faces, key=face_quality, reverse=True)[:120]
    groups = group_similar_faces(best_faces)

    samples = []
    group_list = []
    with lock:
        photo_previews.clear()
        for group_number, group in enumerate(groups, start=1):
            ids = []
            for face in sorted(group, key=lambda f: f["photo"])[:20]:
                image_url = to_data_url(face["crop"])
                if image_url is None:
                    continue
                photo_previews[face["id"]] = face
                ids.append(face["id"])
                samples.append({
                    "id": face["id"],
                    "group": group_number,
                    "photo": face["photo"],
                    "sharpness": face["sharpness"],
                    "image": image_url,
                })
            if ids:
                group_list.append({"id": group_number, "suggested_name": f"Person {group_number}", "ids": ids})

    return jsonify(samples=samples, groups=group_list, count=len(samples))


@app.post("/photo-import/save-groups")
def save_photo_groups():
    """Save the groups the user checked and named."""
    data = request.get_json(silent=True) or {}
    groups = data.get("groups")
    if not isinstance(groups, list) or not 1 <= len(groups) <= MAX_PHOTOS:
        return jsonify(error="Choose at least one group to save."), 400

    # Check everything first, so we never save half of the groups.
    to_save = []
    used_names = set()
    used_ids = set()
    for group in groups:
        if not isinstance(group, dict):
            return jsonify(error="Something is wrong with one of the groups."), 400
        name = str(group.get("name", "")).strip()
        ids = group.get("ids", [])

        if not is_valid_name(name):
            return jsonify(error=f'"{name}" is not a valid name.'), 400
        if not isinstance(ids, list) or not 1 <= len(ids) <= 20:
            return jsonify(error="Every group needs between 1 and 20 faces."), 400
        if name.lower() in used_names:
            return jsonify(error="Give each group a different name."), 400
        for sample_id in ids:
            if not isinstance(sample_id, str) or sample_id in used_ids:
                return jsonify(error="A face can only be in one group."), 400
            used_ids.add(sample_id)

        used_names.add(name.lower())
        to_save.append((name, ids))

    with lock:
        for sample_id in used_ids:
            if sample_id not in photo_previews:
                return jsonify(error="These results are out of date. Please scan the photos again."), 400

        for name, ids in to_save:
            name = find_existing_name(name)
            for sample_id in ids:
                face = photo_previews[sample_id]
                save_sample(name, face["embedding"], face["crop"])
        photo_previews.clear()

    return jsonify(ok=True, people=len(to_save), saved=len(used_ids))


if __name__ == "__main__":
    address = f"http://127.0.0.1:{PORT}"
    print("FaceTracker is running at", address)
    # Open the page in the browser 1 second from now (once the server has started).
    threading.Timer(1, webbrowser.open, [address]).start()
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
