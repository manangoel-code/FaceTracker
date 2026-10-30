"""
Face detection, face recognition and photo grouping.

We use two ready-made OpenCV models (we don't train them ourselves):
  - YuNet finds where the faces are in a picture.
  - SFace turns a face into a list of 128 numbers, called a "face vector".

Two pictures of the same person give face vectors that are very similar,
so we can compare vectors to tell people apart.
"""
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODELS_FOLDER = Path(__file__).resolve().parent / "models"
MODELS_FOLDER.mkdir(exist_ok=True)

DETECTOR_FILE = "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "face_recognition_sface_2021dec.onnx"
DOWNLOAD_URL = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/"

# (file name, folder in the opencv_zoo repository, smallest size the file can be in bytes)
MODEL_FILES = [
    (DETECTOR_FILE, "face_detection_yunet", 100_000),
    (RECOGNIZER_FILE, "face_recognition_sface", 10_000_000),
]


# ---------------------------------------------------------------
# Loading the models
# ---------------------------------------------------------------

def download_missing_models():
    """Download a model file only if it is missing or only half downloaded."""
    for filename, folder, min_size in MODEL_FILES:
        path = MODELS_FOLDER / filename
        if path.exists() and path.stat().st_size >= min_size:
            continue

        print("Downloading", filename, "...")
        temp_path = MODELS_FOLDER / (filename + ".download")
        try:
            urllib.request.urlretrieve(DOWNLOAD_URL + folder + "/" + filename, temp_path)
            if temp_path.stat().st_size < min_size:
                raise ValueError("the downloaded file is too small")
            temp_path.replace(path)
        except Exception as error:
            temp_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not download {filename}: {error}")


def load_models():
    """Return a face detector and a face recognizer, ready to use."""
    download_missing_models()
    detector = cv2.FaceDetectorYN.create(
        str(MODELS_FOLDER / DETECTOR_FILE), "", (320, 320),
        score_threshold=0.82, nms_threshold=0.3, top_k=50,
    )
    recognizer = cv2.FaceRecognizerSF.create(str(MODELS_FOLDER / RECOGNIZER_FILE), "")
    return detector, recognizer


# ---------------------------------------------------------------
# Finding and comparing faces
# ---------------------------------------------------------------

def normalize(vector):
    """Scale a vector so its length is 1. This makes vectors easy to compare."""
    length = np.linalg.norm(vector)
    return vector / max(length, 1e-8)


def similarity(a, b):
    """How alike two face vectors are (cosine similarity).

    1.0 means identical, 0 means unrelated. This is a score,
    not the chance (percentage) that it's the same person.
    """
    return float(np.dot(normalize(a), normalize(b)))


def find_faces(image, detector, recognizer, min_size=1):
    """Find every face in an image.

    Returns a list like:
        [{"box": [x, y, width, height], "embedding": face_vector, "crop": face_image}, ...]
    """
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, detections = detector.detect(image)
    if detections is None:
        return []

    faces = []
    for detection in detections:
        x, y, w, h = [int(value) for value in detection[:4]]
        if w < min_size or h < min_size:
            continue

        # Straighten the face, then turn it into a face vector.
        try:
            aligned = recognizer.alignCrop(image, detection)
            embedding = recognizer.feature(aligned).flatten().astype(np.float32)
        except cv2.error:
            continue

        # Cut the face out of the picture (kept inside the picture edges).
        crop = image[max(0, y):min(height, y + h), max(0, x):min(width, x + w)].copy()
        if crop.size == 0:
            continue

        faces.append({"box": [x, y, w, h], "embedding": normalize(embedding), "crop": crop})
    return faces


def measure_sharpness(image):
    """Bigger number = sharper image. Blurry faces score low."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------------------------------------------------------
# Grouping faces from imported photos
# ---------------------------------------------------------------

def face_quality(face):
    """Big, sharp faces are the most useful, so they get a higher score."""
    width, height = face["box"][2], face["box"][3]
    return width * height * min(face["sharpness"], 180)


def average_face(group):
    """One face vector that represents a whole group."""
    vectors = [face["embedding"] for face in group]
    return normalize(np.mean(vectors, axis=0))


def from_same_photo(group_a, group_b):
    """Two faces in the same photo must be two different people."""
    photos_a = {face["photo"] for face in group_a}
    photos_b = {face["photo"] for face in group_b}
    return len(photos_a & photos_b) > 0


def group_similar_faces(faces, threshold=0.53):
    """Put faces that look alike into groups (one group per suggested person).

    These are only suggestions. The user checks them before anything is saved.
    """
    groups = []

    # Start with the best faces so each group begins with a clear example.
    for face in sorted(faces, key=face_quality, reverse=True):
        best_group = None
        best_score = threshold
        for group in groups:
            if from_same_photo([face], group):
                continue
            score = similarity(face["embedding"], average_face(group))
            closest = max(similarity(face["embedding"], other["embedding"]) for other in group)
            if score > best_score and closest >= 0.48:
                best_group = group
                best_score = score

        if best_group is None:
            groups.append([face])
        else:
            best_group.append(face)

    # Sometimes one person gets split into two groups, so try joining them back.
    joined_something = True
    while joined_something:
        joined_something = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if from_same_photo(groups[i], groups[j]):
                    continue
                if similarity(average_face(groups[i]), average_face(groups[j])) >= 0.64:
                    groups[i].extend(groups.pop(j))
                    joined_something = True
                    break
            if joined_something:
                break

    # Biggest groups first.
    groups.sort(key=lambda group: (len(group), max(face_quality(f) for f in group)), reverse=True)
    return groups
