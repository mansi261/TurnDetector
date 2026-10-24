# TurnDetector

## Clone and run

### Prerequisites

- Git
- Docker Desktop, or Docker Engine with Docker Compose v2
- At least 2 GB of free disk space for the image and build cache

The model runs locally and does not require an API key.

### Start the service

```bash
git clone https://github.com/mansi261/TurnDetector.git
cd TurnDetector
docker compose up --build -d
```

The first build downloads the Python base image and installs the pinned serving
dependencies. Later starts reuse Docker's build cache.

Wait until the model is ready:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
```

Then use:

- Demo and API dashboard: `http://127.0.0.1:8000`
- API metrics: `http://127.0.0.1:8000/metrics`

The demo model dropdown selects either **English** or **Multilingual — 14
languages** before the microphone session starts. The selected model remains
fixed for that session.

To call the model with a 16 kHz WAV or FLAC file:

```bash
curl --fail --request POST \
  --form "audio=@/absolute/path/to/audio.wav" \
  http://127.0.0.1:8000/predict
```

Add `?model=multilingual` to use the multilingual artifact. English is the
default.

The response contains `p_end`, the `hold` or `end` decision, selected model ID,
threshold, model version, and preprocessing/inference timing.

View container logs or stop the service with:

```bash
docker compose logs --follow
docker compose down
```

If port 8000 is already in use, select another host port:

```bash
TURN_DETECTOR_PORT=8001 docker compose up --build -d
```

In that case, open `http://127.0.0.1:8001` and use port 8001 for the API calls.
