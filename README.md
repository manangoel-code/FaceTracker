# FaceTracker

A small Python app that recognizes people through a webcam. It runs entirely on your own computer.

Save a few face samples of someone and the app labels them on the live camera from then on. You can also import photos and let the app group the faces it finds, and a benchmark page measures how well recognition works and exports the results as CSV.

## How to run it

You need [Python](https://www.python.org/downloads/) 3.10 or newer.

On Windows, double-click `run.bat`. The first run sets everything up, which takes a minute or two, then the app opens in your browser. Close the black window to stop it.

To start it by hand instead, open Command Prompt in this folder and run:

```bat
py -m pip install -r requirements.txt
py app.py
```

The page opens at http://127.0.0.1:5000. The first time you start the camera, the app downloads its two face models into the `models` folder.

To check that nothing is broken after a change, run `py test_app.py`.

## Files

| File | What it does |
| --- | --- |
| `run.bat` | Sets up and starts the app |
| `app.py` | The web server: webcam, saved people, benchmark and photo import |
| `vision.py` | Loads the face models, finds faces and groups similar ones |
| `metrics.py` | Works out the benchmark numbers (accuracy, precision, recall) |
| `test_app.py` | Quick checks for names, recognition, benchmark numbers and access rules |
| `templates/index.html` | The page layout |
| `static/app.js` | What the buttons do |
| `static/style.css` | How the page looks (colours are at the top) |
| `saved_samples/` | One folder per person with their face pictures and face vectors |
| `settings.json` | Your recognition settings, written when you press "Apply and save" |

## How it works

YuNet, an OpenCV model, finds the faces in each frame. A second OpenCV model, SFace, turns each face into 128 numbers (a face vector). The app compares that vector with the saved ones and shows a name only if the face is similar enough to that person and clearly closer to them than to anyone else.

## Tips

- Press "Take 10 samples" and turn your head slowly while it runs. A second round in different lighting helps.
- Delete blurry or wrong samples on the People page.
- The similarity percentage is not the chance that the name is right.
- Check photo groups before saving them. Bad lighting, side-on faces and blur all cause mistakes.

## Privacy

Only save the faces of people who agreed to it. Saved samples stay on your computer, and `.gitignore` keeps them out of Git.
