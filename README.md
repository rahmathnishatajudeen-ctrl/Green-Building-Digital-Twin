# Green Building Digital Twin

Detect visible people with pretrained YOLO11n, assign them to room zones, and emit
timestamped JSONL observations for the next digital-twin layer. No facial
recognition, identity tracking, image recording, or cloud inference is used.

## Run on Windows (PowerShell)

Python 3.10+ is required. The workspace already has an installed `.venv`.
For a fresh checkout:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
```

Webcam, with local preview (press Q to stop):

```powershell
.\.venv\Scripts\python.exe -m gbdt_vision.cli --source 0 --preview
```

Recorded video with two example room zones:

```powershell
.\.venv\Scripts\python.exe -m gbdt_vision.cli --source "C:\videos\office.mp4" --zones config/zones.example.json --preview
```

Headless processing:

```powershell
.\.venv\Scripts\python.exe -m gbdt_vision.cli --source "C:\videos\office.mp4" --every 10 --output data/observations.jsonl
```

Weights download on first run; subsequent inference is local. `--model` accepts a
local checkpoint for offline use. CPU is the default; `--device 0` requires a
compatible CUDA/PyTorch installation. `--max-frames 10` limits analyzed frames.
Video is processed as fast as inference allows, not at playback speed.

## Configure camera zones

Without a configuration, the whole frame is `room-1`. Edit the example to match
the actual camera view. Bounds are normalized `[left, top, right, bottom]` in
the range 0 to 1, with the origin at the upper-left corner.

Each person's bounding-box bottom-center determines their room. Shared edges
count once; overlapping zones use the first matching zone. Room IDs must be
unique per camera. The example's left/right split is illustrative, not a learned
room layout. Do not sum overlapping camera views without cross-camera deduplication.

Full-frame brightness is measured by default. Optionally add
`"lighting_bounds": [0.1, 0.1, 0.2, 0.2]` to a room to measure a fixed region
instead. Output is 0 to 1, **not lux, light
switch state, or energy consumption**. Daylight, auto-exposure and camera changes
affect brightness; use a calibrated sensor for reliable lighting decisions.

## Output contract

Each analyzed frame appends one JSON object to `data/observations.jsonl`:

```json
{
  "schema_version": "1.0",
  "status": "observed",
  "observed_at": "2026-10-15T12:00:00+00:00",
  "camera_id": "camera-1",
  "source_type": "video",
  "frame_index": 0,
  "video_offset_seconds": 0.0,
  "frame_size": {"width": 1280, "height": 720},
  "total_visible_people": 2,
  "unassigned_people": 0,
  "rooms": [
    {"room_id": "room-1", "occupancy_count": 2,
    "lighting": {"status": "observed", "mean_brightness": 0.5, "scope": "full_frame"}}
  ],
  "model": "yolo11n.pt",
  "confidence_threshold": 0.4
}
```

`observed_at` is UTC processing time, not the original recording date. Recorded
video also includes its frame-based timeline offset when FPS is available.
Only successful inference emits occupancy observations. Camera failure exits
nonzero rather than emitting zero occupancy; downstream consumers must treat
stale data as unknown. Normal video EOF stops processing. Some codecs cannot
distinguish mid-file corruption from EOF. Output appends across runs.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover zone boundaries, overlap, unassigned detections, camera failure,
brightness, JSON serialization and frame sampling using a generated video.
Model accuracy must still be evaluated on consented footage from the actual
building. Occlusion, reflections, camera placement and missed detections affect
counts. A visible-person count is an estimate, not proof that a room is empty.
Do not automatically disable ventilation or safety systems based on this output.

Use authorized cameras and footage. Preview shows identifiable video locally;
disable it in privacy-sensitive settings. JSONL still contains occupancy data:
restrict access and define a retention policy. Review Ultralytics' AGPL-3.0 and
enterprise licensing terms before distributing or commercializing the solution.

## Floor-plan twin

The second layer displays camera observations on a selectable floor plan, with a
room register, capacity indicators, camera mappings and data freshness. Room
details show relative brightness and its lighting scenario. Temperature, power and CO2 are
explicitly not connected; these are not simulated in live mode.

The dashboard dependencies are already installed in the workspace. For a fresh
checkout, install the optional dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[twin,ai,test]"
```

Start the dashboard from the project root:

```powershell
.\.venv\Scripts\python.exe -m gbdt_vision.twin_server --port 8000
```

Open http://127.0.0.1:8000 for live data, or
http://127.0.0.1:8000/?mode=demo for clearly labeled simulated observations. Demo
mode never writes into the live observation file. Select rooms on the plan or in
the register; the dashboard polls every three seconds. Keyboard selection uses
Tab and Enter/Space.

Run vision in a separate terminal to populate the default workspace:

```powershell
.\.venv\Scripts\python.exe -m gbdt_vision.cli --source 0 --camera-id camera-1 --preview
```

The default vision configuration reports `room-1`, mapped to Open workspace.
Using `--zones config/zones.example.json` instead reports `meeting-room-a` and
`work-area`, mapped to the other two spaces. Only run the configuration that
matches the actual view. Unobserved rooms remain unknown or stale; they are not
filled with demo values. Avoid reusing a camera ID for unrelated camera views.

### Floor configuration

Edit `config/floor-plan.json` to set building metadata and room geometry. The
provided floor is an illustrative three-room layout, not an inferred building
plan. Geometry uses floor-local drawing units (`x`, `y`, `width`, `height`), not
normalized camera coordinates. Camera zones and floor geometry are separate:
each room binds one `camera_id` + `vision_room_id` to a physical room. These
bindings must be unique. Capacity is configured metadata, not a measured value.

Use custom paths or a different freshness threshold with:

```powershell
.\.venv\Scripts\python.exe -m gbdt_vision.twin_server --floor-plan config/floor-plan.json --observations data/observations.jsonl --stale-seconds 15 --port 8001
```

The default reader scans up to the last 2 MB of JSONL per request, skips malformed
records and incomplete trailing lines, and chooses the newest valid UTC timestamp
for each binding. This bounded reader is suitable for the hackathon; a production
ingestion service should retain the latest state per source in a database.

After 15 seconds without an observation, current occupancy becomes unknown; the
last-known count remains visible separately. A browser-side expiry guard also
handles stalled connections. Room coverage indicates how much of the floor
contributes to the visible-person total. This total is not a building-wide census
when rooms are unobserved. Do not map overlapping camera views to separate rooms
and add their counts. Recorded-video inputs remain labeled as recorded video;
freshness refers to processing time, not when footage was captured.

The API is available at `/api/twin` (or `/api/twin?mode=demo`), with interactive
API documentation at `/docs`. The service binds to localhost by default and has
no authentication. Do not expose it externally without authentication, TLS and
an appropriate data-access policy. Dashboard fonts use Google Fonts, with local
fallbacks when offline; occupancy data is not sent to that font service.

## Lighting scenario

Lighting scenarios now use **frame brightness**, replacing the previous
vacancy/light-switch waste scenario. Mean OpenCV grayscale intensity is divided
by 255 to obtain relative camera brightness from 0 to 1. Without extra
configuration, the entire frame is measured on each analyzed frame.

| Scenario | Default brightness | Dashboard message |
| --- | --- | --- |
| Low lighting | Below 30% | The scene is too dark. |
| Normal lighting | 30% through 75%, inclusive | The scene has normal lighting. |
| Excessive lighting | Above 75% | The scene is too bright. |

The dashboard shows scenario counts, a message and brightness percentage for
each room, a selected-room brightness meter, and labels on the floor plan and
room register. Classification does not depend on occupancy, vacancy duration or
a separate light-state sensor. Missing or stale frames show **Unknown lighting**,
never Normal. Camera brightness does not establish light-switch state or energy
consumption, so the previous switch-off controls and kWh estimates are removed.

Run the live dashboard and vision in separate terminals:

```powershell
.\.venv\Scripts\python.exe -m gbdt_vision.twin_server --port 8000
.\.venv\Scripts\python.exe -m gbdt_vision.cli --source 0 --preview
```

Restart an already-running vision process to enable default full-frame brightness.
Open http://127.0.0.1:8000 and select **Live feed**. Recorded video is supported as
well, with its source labeled as recorded video. For a room-specific region,
adjust `lighting_bounds` in `config/zones.lighting.example.json` and pass that
file using `--zones`. Without a region, rooms using the same camera share its
full-frame brightness; these are not independent room-level measurements.

Each room's `lighting` settings in `config/floor-plan.json` can override
`low_brightness_threshold` (default 0.30) and `high_brightness_threshold` (default
0.75). Values must satisfy `0 <= low < high <= 1`. Restart the twin server after
changing the configuration. These are heuristic camera thresholds, not lux
standards; tune them for your scene and camera exposure. Daylight, auto-exposure
and scene colors affect readings. An illuminated bright surface can raise the
mean even if the rest of a room is dark.

Demo mode at http://127.0.0.1:8000/?mode=demo shows an Open workspace at 15%
(Low), Meeting room A at 90% (Excessive), and Work area at 50% (Normal). Demo
values are simulated and do not change live files. The legacy
`POST /api/lighting/readings` sensor ingestion endpoint remains available for
compatibility but is not used by brightness classification. The old demo
switch-off/reset endpoints are no longer part of this scenario.

Tests cover all three scenarios, threshold boundaries, invalid and stale values,
full-frame and regional measurement, live API updates and demo isolation, in
addition to the occupancy and floor-plan checks.

## GenAI recommendations

The Recommendations panel takes **number of people** and **lighting condition**
(Low, Normal or Excessive). Use **Room data** for the selected room's current
observations, or **What-if** to enter a manual scenario. **Generate recommendations**
returns a summary, up to three actions, the exact input basis and a safety caveat.
Live, recorded-video, demo and manual inputs are labeled separately. Old advice
is cleared if its inputs change or the room data expires.

By default, the service uses an explicitly labeled **rule-based fallback**, not
GenAI. It works without credentials. To enable actual model generation, install
the optional SDK (already installed in this workspace):

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[twin,ai,test]"
```

### OpenAI

Set `OPENAI_API_KEY` securely in the server process environment. Do not put keys
in the browser, source code, commits or chat. Then start the server in that same
terminal with the non-secret configuration:

```powershell
$env:GBDT_AI_PROVIDER = "openai"
$env:OPENAI_MODEL = "gpt-4o-mini"
.\.venv\Scripts\python.exe -m gbdt_vision.twin_server --port 8000
```

Use a model available to your account that supports Chat Completions JSON mode.
Model names can be changed with `OPENAI_MODEL`.

### Azure OpenAI

Set `AZURE_OPENAI_API_KEY` securely in the server process environment. Configure
the resource endpoint and your deployed model name before starting:

```powershell
$env:GBDT_AI_PROVIDER = "azure"
$env:AZURE_OPENAI_ENDPOINT = "https://YOUR-RESOURCE.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "YOUR-DEPLOYMENT-NAME"
$env:AZURE_OPENAI_API_VERSION = "2024-10-21"
.\.venv\Scripts\python.exe -m gbdt_vision.twin_server --port 8000
```

Choose an API version and deployment supporting Chat Completions JSON mode.
Stop an existing server before restarting on the same port. Environment settings
are inherited at startup; refresh the dashboard after a restart. `.env` files
are not automatically loaded. Set `GBDT_AI_PROVIDER=rules` to disable cloud AI.

### Behavior and limitations

- Cloud requests send only the people count, lighting category and grounded
  lighting guidance. No video, images, identities, room names or building plans
  are sent. Counts still reveal usage patterns; use an approved provider.
- Generation is on demand, never part of the three-second dashboard polling.
  Successful AI results are cached by input combination for five minutes, with
  a maximum of 128 entries. Cached results are labeled.
- The SDK uses a 12-second timeout and no automatic retries. Provider failures,
  refusals and malformed responses use the labeled rule-based fallback; failed
  providers have a 60-second cooldown. Error responses do not expose credentials
  or raw provider errors. Configuration status does not prove credentials work.
- Advice is grounded in the six occupied/unoccupied brightness combinations.
  An empty dark room does not need extra lighting just to brighten the camera.
  Occupied rooms prioritize task visibility, glare control and targeted lighting.
- Brightness is not lux or switch state. Advice does not claim quantified energy,
  cost or carbon savings. AI wording is advisory and still requires operator
  review. No physical controls are connected or invoked.

`POST /api/recommendations` accepts manual inputs, for example:

```json
{"mode":"manual","people_count":5,"lighting_condition":"low"}
```

For observed data, send `{"mode":"live","room_id":"workspace"}` or use
`"mode":"demo"`. The server reads the actual room inputs instead of trusting
client counts, and rechecks freshness and changes after generation. Unknown or
stale data returns HTTP 409 rather than guessing. `GET /api/recommendations/status`
reports the configured provider without secrets. The local service still needs
authentication, rate limits and TLS before exposure outside localhost.

Tests cover grounding, input validation, mock OpenAI/Azure configuration and
generation, output validation, fallback, caching, stale/changed room rejection
and the separation between polling and on-demand generation. Real external
model output must be verified with your authorized credentials and deployment.