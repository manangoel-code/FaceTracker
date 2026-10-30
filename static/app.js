// FaceTracker - what the buttons on the page do.
// The page talks to app.py by sending requests like "/camera/start" or "/status".


// ===============================================================
// Small helpers
// ===============================================================

// Shortcut: $("name") instead of document.getElementById("name")
function $(id) {
  return document.getElementById(id);
}

// Send a request to app.py and get the answer back.
// If app.py replies with an error, this throws it so a catch block can show it.
async function api(path, method = "GET", data) {
  const options = { method: method };
  if (data !== undefined) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(data);
  }
  const response = await fetch(path, options);
  const answer = await response.json();
  if (!response.ok) {
    throw new Error(answer.error || "Something went wrong.");
  }
  return answer;
}

// Make text safe to put inside HTML (so a name can't break the page).
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = String(text);
  return div.innerHTML;
}

// Show a small message at the bottom of the screen for a few seconds.
let toastTimer = null;
function showMessage(text, isError = false) {
  const toast = $("toast");
  toast.textContent = text;
  toast.className = isError ? "toast error" : "toast";
  toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toast.hidden = true; }, 3500);
}

function showError(error) {
  showMessage(error.message || String(error), true);
}

// Fill a <select> with options, keeping the current choice if it still exists.
// options is a list of [value, text] pairs.
function fillSelect(select, options) {
  const current = select.value;
  select.innerHTML = "";
  for (const [value, text] of options) {
    select.add(new Option(text, value));
  }
  if (options.some(([value]) => String(value) === current)) {
    select.value = current;
  }
}


// ===============================================================
// Tabs
// ===============================================================

function showTab(name) {
  for (const page of ["camera", "people", "benchmark", "help"]) {
    $("page-" + page).hidden = page !== name;
  }
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.tab === name);
  }
  if (name === "benchmark") {
    loadBenchmarkResults();
  }
}


// ===============================================================
// Live camera
// ===============================================================

async function startCamera() {
  try {
    await api("/camera/start", "POST");
    // Adding the time to the address makes the browser open a fresh stream.
    $("video").src = "/video?t=" + Date.now();
    $("video").style.display = "block";
    $("offline").hidden = true;
  } catch (error) {
    showError(error);
  }
}

async function stopCamera() {
  try {
    await api("/camera/stop", "POST");
    $("video").removeAttribute("src");
    $("video").style.display = "none";
    $("offline").hidden = false;
    $("offlineText").textContent = "Camera is off";
  } catch (error) {
    showError(error);
  }
}

async function saveSettings() {
  try {
    await api("/settings", "POST", {
      threshold: Number($("threshold").value),
      margin: Number($("margin").value),
      width: Number($("width").value),
      detection_every: Number($("every").value),
    });
    showMessage("Settings applied.");
  } catch (error) {
    showError(error);
  }
}

// Save the chosen face once. Returns {name, samples} from app.py.
function saveOneSample() {
  return api("/enroll", "POST", {
    name: $("name").value.trim(),
    face_id: $("face").value,
  });
}

// Wait a number of milliseconds (use it with await).
function wait(milliseconds) {
  return new Promise(resolve => setTimeout(resolve, milliseconds));
}

async function enroll() {
  try {
    const result = await saveOneSample();
    showMessage(`Saved sample ${result.samples} for ${result.name}.`);
    await refresh();
  } catch (error) {
    showError(error);
  }
}

// Take 10 samples, one every half second, while the person turns their head.
let capturing = false;

async function autoCapture() {
  if (capturing) return;  // already running
  capturing = true;
  const status = $("captureStatus");

  try {
    for (let i = 1; i <= 10; i++) {
      await saveOneSample();
      status.textContent = `${i} of 10 saved. Keep turning your head slowly...`;
      await wait(500);
    }
    showMessage("Done! 10 samples saved.");
  } catch (error) {
    showError(error);
  }

  status.textContent = "";
  capturing = false;
  await refresh();
}


// ===============================================================
// Keeping the page up to date
// ===============================================================
// Every second we ask app.py for the latest status and update the page.

let lastPeopleList = "";
let settingsLoaded = false;

async function refresh() {
  let status;
  try {
    status = await api("/status");
  } catch (error) {
    $("cameraStatus").textContent = "app not running";
    $("cameraStatus").className = "camera-status error";
    return;
  }

  updateCameraStatus(status);
  updateFaceChoices(status.faces);

  // Fill the settings boxes once, with the values app.py is using.
  if (!settingsLoaded) {
    $("threshold").value = status.settings.threshold;
    $("margin").value = status.settings.margin;
    $("width").value = status.settings.width;
    $("every").value = status.settings.detection_every;
    settingsLoaded = true;
  }

  // Only redraw the people list when it has changed.
  const peopleText = JSON.stringify(status.people);
  if (peopleText !== lastPeopleList) {
    lastPeopleList = peopleText;
    showPeople(status.people);
  }

  if (status.benchmark.active) {
    $("benchmarkStatus").textContent =
      `Recording... ${status.benchmark.samples} frames so far. ` +
      `In view: ${status.benchmark.expected || "nobody saved"}`;
    if (!$("page-benchmark").hidden) {
      loadBenchmarkResults(false);
    }
  }
}

function updateCameraStatus(status) {
  const label = $("cameraStatus");
  if (status.error) {
    label.textContent = "camera problem";
    label.className = "camera-status error";
    $("offline").hidden = false;
    $("video").style.display = "none";
    $("offlineText").textContent = status.error;
  } else if (status.running) {
    label.textContent = "camera on";
    label.className = "camera-status on";
  } else if ($("video").getAttribute("src")) {
    label.textContent = "starting...";
    label.className = "camera-status";
  } else {
    label.textContent = "camera off";
    label.className = "camera-status";
  }
  // Only show "LIVE" when the video is actually showing on this page.
  const showingVideo = Boolean($("video").getAttribute("src"));
  $("liveTag").hidden = !(status.running && showingVideo);
  if (!showingVideo && !status.error) {
    $("offlineText").textContent = status.running ? "Camera is on in another tab" : "Camera is off";
  }

  if (status.running) {
    $("fpsValue").textContent = status.fps;
    $("latencyValue").textContent = status.latency_ms;
    $("faceValue").textContent = status.faces.length;
  } else {
    $("fpsValue").textContent = "-";
    $("latencyValue").textContent = "-";
    $("faceValue").textContent = "-";
  }
}

function updateFaceChoices(faces) {
  const options = [["-1", "The only face in view"]];
  for (const face of faces) {
    const label = face.name ? `Face ${face.id} (${face.name})` : `Face ${face.id}`;
    options.push([String(face.id), label]);
  }
  fillSelect($("face"), options);
}


// ===============================================================
// Saved people
// ===============================================================

let openPerson = null;  // whose samples are being shown right now

function showPeople(people) {
  $("peopleCount").textContent = people.length;

  const list = $("peopleList");
  if (people.length === 0) {
    list.innerHTML = `<p class="muted">No one saved yet. Add someone on the Camera page.</p>`;
  } else {
    list.innerHTML = "";
    for (const person of people) {
      const row = document.createElement("div");
      row.className = "person";
      row.innerHTML = `
        <span class="person-name">${escapeHtml(person.name)}</span>
        <span class="person-samples">${person.samples} sample${person.samples === 1 ? "" : "s"}</span>
        <button class="plain small-button">Samples</button>
        <button class="plain small-button">Rename</button>
        <button class="danger small-button">Delete</button>`;
      const buttons = row.querySelectorAll("button");
      buttons[0].onclick = () => openSamples(person.name);
      buttons[1].onclick = () => renamePerson(person.name);
      buttons[2].onclick = () => deletePerson(person.name);
      list.append(row);
    }
  }

  // The "Who is in view?" list on the Benchmark page uses the same people.
  const choices = [["None", "Nobody saved"]];
  for (const person of people) {
    choices.push([person.name, person.name]);
  }
  fillSelect($("expected"), choices);
}

function personUrl(name) {
  return "/people/" + encodeURIComponent(name);
}

async function openSamples(name) {
  try {
    const result = await api(personUrl(name) + "/samples");
    openPerson = name;
    $("samplesTitle").textContent = `${name}'s samples`;
    $("samplesCard").hidden = false;

    const grid = $("samplesGrid");
    grid.innerHTML = "";
    if (result.samples.length === 0) {
      grid.textContent = "No samples saved.";
    }
    for (const sample of result.samples) {
      const item = document.createElement("div");
      item.className = "face";
      if (sample.has_image) {
        item.innerHTML = `<img src="${personUrl(name)}/samples/${sample.id}/image" alt="Face sample">`;
      } else {
        item.innerHTML = `<div class="no-image">No picture</div>`;
      }
      const button = document.createElement("button");
      button.className = "danger small-button";
      button.textContent = "Delete";
      button.onclick = () => deleteSample(name, sample.id);
      item.append(button);
      grid.append(item);
    }
  } catch (error) {
    showError(error);
  }
}

function closeSamples() {
  openPerson = null;
  $("samplesCard").hidden = true;
}

async function deleteSample(name, sampleId) {
  if (!confirm("Delete this face sample?")) return;
  try {
    await api(personUrl(name) + "/samples/" + sampleId, "DELETE");
    await openSamples(name);
    await refresh();
  } catch (error) {
    showError(error);
  }
}

async function renamePerson(name) {
  const newName = prompt("New name:", name);
  if (newName === null) return;
  try {
    await api(personUrl(name), "PATCH", { name: newName.trim() });
    showMessage(`Renamed ${name} to ${newName.trim()}.`);
    await refresh();
    if (openPerson === name) {
      await openSamples(newName.trim());
    }
  } catch (error) {
    showError(error);
  }
}

async function deletePerson(name) {
  if (!confirm(`Delete ${name} and all their samples?`)) return;
  try {
    await api(personUrl(name), "DELETE");
    if (openPerson === name) closeSamples();
    showMessage(`Deleted ${name}.`);
    await refresh();
  } catch (error) {
    showError(error);
  }
}


// ===============================================================
// Importing photos
// ===============================================================
// The groups are only suggestions. The user can rename them, move faces
// between groups, or leave faces out before saving.

let photoGroups = [];   // [{id: 1, name: "Person 1", save: true}, ...]
let photoFaces = [];    // [{id: "...", group: 1, photo: 2, sharpness: 80, image: "data:..."}, ...]

async function previewPhotos() {
  const status = $("photoStatus");
  const files = $("photoFiles").files;

  if (files.length === 0 || files.length > 30) {
    status.textContent = "Choose between 1 and 30 photos.";
    return;
  }
  if (!$("photoPermission").checked) {
    status.textContent = "Please tick the permission box first.";
    return;
  }

  const form = new FormData();
  for (const file of files) {
    form.append("photos", file);
  }
  form.append("permission", "yes");

  status.textContent = "Looking for faces... this can take a few seconds.";
  try {
    const response = await fetch("/photo-import/preview", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not read the photos.");

    photoFaces = result.samples;
    photoGroups = result.groups.map(group => ({ id: group.id, name: group.suggested_name, save: true }));
    showGroups();
    status.textContent = `Found ${result.count} faces in ${result.groups.length} groups. Check them below.`;
  } catch (error) {
    status.textContent = error.message;
  }
}

function showGroups() {
  $("reviewCard").hidden = false;
  const list = $("groupList");
  list.innerHTML = "";

  if (photoFaces.length === 0) {
    list.textContent = "No clear faces were found in these photos.";
    return;
  }

  for (const group of photoGroups) {
    const faces = photoFaces.filter(face => face.group === group.id);

    const card = document.createElement("div");
    card.className = "group";
    card.innerHTML = `
      <div class="group-top">
        <label>Name <input maxlength="50"></label>
        <label class="checkbox"><input type="checkbox"> Save</label>
        <span class="muted small">${faces.length} face${faces.length === 1 ? "" : "s"}</span>
      </div>
      <div class="face-grid"></div>`;

    const nameBox = card.querySelector("input");
    nameBox.value = group.name;
    nameBox.oninput = () => { group.name = nameBox.value; };

    const saveBox = card.querySelector("input[type=checkbox]");
    saveBox.checked = group.save;
    saveBox.onchange = () => { group.save = saveBox.checked; };

    const grid = card.querySelector(".face-grid");
    for (const face of faces) {
      grid.append(makeFaceTile(face));
    }
    list.append(card);
  }
}

// One face picture with a menu to move it to another group.
function makeFaceTile(face) {
  const tile = document.createElement("div");
  tile.className = "face";
  tile.innerHTML = `
    <img src="${face.image}" alt="Face from photo ${face.photo}">
    <small>Photo ${face.photo} &middot; sharpness ${Math.round(face.sharpness)}</small>
    <select aria-label="Move to group"></select>`;

  const menu = tile.querySelector("select");
  const choices = photoGroups.map(group => [String(group.id), "Group " + group.id]);
  choices.push(["0", "Leave out"]);
  fillSelect(menu, choices);
  menu.value = String(face.group);
  menu.onchange = () => {
    face.group = Number(menu.value);
    showGroups();
  };
  return tile;
}

async function saveGroups() {
  const groups = [];
  for (const group of photoGroups) {
    const ids = photoFaces.filter(face => face.group === group.id).map(face => face.id);
    if (group.save && ids.length > 0) {
      groups.push({ name: group.name.trim(), ids: ids });
    }
  }

  if (groups.length === 0) {
    showMessage("Tick at least one group to save.", true);
    return;
  }

  const faceCount = groups.reduce((total, group) => total + group.ids.length, 0);
  if (!confirm(`Save ${groups.length} people with ${faceCount} face samples?`)) return;

  try {
    const result = await api("/photo-import/save-groups", "POST", { groups: groups });
    showMessage(`Saved ${result.saved} samples for ${result.people} people.`);
    photoGroups = [];
    photoFaces = [];
    $("reviewCard").hidden = true;
    $("photoStatus").textContent = "Saved. You can import more photos.";
    await refresh();
  } catch (error) {
    showError(error);
  }
}


// ===============================================================
// Benchmark
// ===============================================================

function testSetup() {
  return {
    expected: $("expected").value,
    lighting: $("lighting").value,
    movement: $("movement").value,
    distance: $("distance").value,
    notes: $("testNotes").value,
  };
}

async function startBenchmark() {
  try {
    await api("/benchmark/start", "POST", testSetup());
    showMessage("Test started.");
    await loadBenchmarkResults();
  } catch (error) {
    showError(error);
  }
}

async function updateBenchmark() {
  try {
    await api("/benchmark/mark", "POST", testSetup());
    showMessage("Test updated.");
  } catch (error) {
    showError(error);
  }
}

async function stopBenchmark() {
  try {
    await api("/benchmark/stop", "POST");
    showMessage("Test stopped.");
    await loadBenchmarkResults();
  } catch (error) {
    showError(error);
  }
}

function asPercent(value) {
  return value === null ? "N/A" : value + "%";
}

async function loadBenchmarkResults(updateStatusText = true) {
  try {
    const result = await api("/benchmark/summary");
    const cards = $("resultCards");
    const outcomes = $("outcomeList");

    if (!result.samples) {
      cards.innerHTML = "";
      outcomes.innerHTML = "";
      if (updateStatusText) $("benchmarkStatus").textContent = "No test recorded yet.";
      return;
    }
    if (updateStatusText) {
      $("benchmarkStatus").textContent = `${result.samples} frames recorded.`;
    }

    const numbers = [
      ["Accuracy", asPercent(result.accuracy)],
      ["Precision", asPercent(result.precision)],
      ["Recall", asPercent(result.recall)],
      ["Average FPS", result.avg_fps],
      ["Avg. processing", result.avg_latency_ms + " ms"],
      ["Slowest 5%", result.p95_latency_ms + " ms"],
      ["False matches", result.false_matches],
      ["Missed faces", result.missed_faces],
    ];
    cards.innerHTML = "";
    for (const [label, value] of numbers) {
      cards.innerHTML += `<div class="result"><b>${value}</b><small>${label}</small></div>`;
    }

    const outcomeNames = {
      correct_identification: "Right person",
      correct_rejection: "Correctly empty",
      wrong_identity: "Wrong person",
      missed_detection: "Face not found",
      unknown_face: "Face not recognized",
      false_detection: "Face found when nobody expected",
    };
    outcomes.innerHTML = "";
    for (const [key, label] of Object.entries(outcomeNames)) {
      outcomes.innerHTML += `<div><span>${label}</span><b>${result.outcomes[key] || 0}</b></div>`;
    }
  } catch (error) {
    showError(error);
  }
}


// ===============================================================
// Start
// ===============================================================

refresh();
setInterval(refresh, 1000);
