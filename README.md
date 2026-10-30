# FaceTracker

A small Python app that recognizes people through your webcam. Everything runs on your own computer.

- **Camera** - draws a box around every face and shows the name if it knows the person
- **People** - see, rename and delete saved people and their face samples
- **Import from photos** - find faces in your photos, group the ones that look alike and save them
- **Benchmark** - test how well recognition works and download the results as a CSV file

## How to run it

You need [Python](https://www.python.org/downloads/) 3.10 or newer.

**Easy way:** double-click `run.bat`. The first time, it sets everything up, which takes a minute or two. Then the app opens in your browser. Close the black window to stop it.

**Or by hand:** open Command Prompt in this folder and run:

```bat
py -m pip install -r requirements.txt
py app.py
```

The page opens by itself at http://127.0.0.1:5000.

The first time you start the camera, the app downloads the two face models into the `models` folder if they aren't there yet.

## What's in each file

| File | What it does |
| --- | --- |
| `run.bat` | Double-click to set up and start the app |
| `app.py` | The web server: webcam, saved people, benchmark and photo import |
| `vision.py` | Loads the face models, finds faces and groups similar ones |
| `metrics.py` | Works out the benchmark numbers (accuracy, precision, ...) |
| `templates/index.html` | The layout of the page |
| `static/app.js` | What happens when you click the buttons |
| `static/style.css` | How the page looks (colours are at the top) |
| `saved_samples/` | One folder per person with their face pictures and face vectors |
| `settings.json` | Your recognition settings (made when you first press "Apply and save") |

## How it works

1. **YuNet** (an OpenCV model) finds where the faces are in each frame.
2. **SFace** (another OpenCV model) turns each face into a list of 128 numbers, called a face vector.
3. The app compares that vector with the saved ones. If it's similar enough to someone, and clearly more similar than to anyone else, their name is shown.

## Tips

- Use **Take 10 samples** and slowly turn your head while it runs. Doing it again in different lighting helps too.
- Delete blurry or wrong samples on the Saved People page.
- The similarity number is not the chance that the name is right.
- Lighting, side-on faces and blur can all cause mistakes, so check photo groups before saving them.

## Privacy

Only save the faces of people who have said yes. Your saved samples stay on your computer and are ignored by `.gitignore`, so they won't be uploaded if you put this project on GitHub.
