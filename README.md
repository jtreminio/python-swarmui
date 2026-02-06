# Python SwarmUI Clone

Single-user Python application that mirrors SwarmUI's core local behavior:
- ComfyUI self-start backend orchestration
- model/wildcard mount integration
- text-to-image generation queue
- web UI + JSON API

## Scope
Included:
- Single-user access (no auth/roles)
- Swarm-style Comfy startup pattern (`main.py` + dynamic port + GPU env vars + generated `comfy-auto-model.yaml`)
- Default txt2img workflow submission to Comfy API
- Output gallery and job tracking
- Swarm-compatible API surface under `/API/{Call}` for key generation/status/model/image calls

Excluded:
- Swarm custom extensions (`src/Extensions`)
- Multi-user permissions
- Shipping a bundled ComfyUI backend

## Default mount assumptions
- Standalone ComfyUI root: `/Users/your-user/comfyui` (example)
- Model root: `/Volumes/models`
- Wildcards root: `/Volumes/aiconfigs/Wildcards`
- Default Comfy start script: `comfyui/main.py` (project-relative)

## Run
1. (Recommended) Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Bootstrap dependencies into the current Python environment (app + Comfy requirements, checks system deps):

```bash
python setup.py
```

3. Start server (uses `config.yaml` host/port):

```bash
python run.py
```

Manual alternative:

```bash
pip install .
uvicorn swarmui_clone.main:app --host 0.0.0.0 --port 7801 --reload
```

4. Open locally at [http://127.0.0.1:7801/api/](http://127.0.0.1:7801/api/) or from LAN at `http://<your-machine-ip>:7801/api/`.

## API quick reference
- `GET /api/health`
- `GET /api/status`
- `GET /api/settings`
- `PUT /api/settings`
- `POST /api/backend/start`
- `POST /api/backend/stop`
- `POST /api/backend/restart`
- `GET /api/backend/preflight` (dependency/start-script preflight diagnostics)
- `GET /api/models`
- `GET /api/wildcards`
- `POST /api/generate`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/cancel`
- `GET /api/output`
- `GET /api/image/{relative_path}`
- `ANY /api/comfy/{path}` proxy to live Comfy backend

Swarm compatibility routes:
- `GET|POST /API/GetCurrentStatus`
- `GET|POST /API/ListT2IParams`
- `GET|POST /API/GenerateText2Image`
- `WS /API/GenerateText2ImageWS`
- `GET|POST /API/ListImages`
- `GET|POST /API/ListModels`
- `GET|POST /API/DescribeModel`
- `GET|POST /API/ListLoadedModels`
- `GET|POST /API/SelectModel`
- `WS /API/SelectModelWS`
- `GET|POST /API/GetModelHash`
- `GET|POST /API/ForwardMetadataRequest`
- `GET|POST /API/TriggerRefresh`
- `GET|POST /API/InterruptAll`
- `GET /View/{relative_path}`
- `GET /Output/{relative_path}`
- `ANY /ComfyBackendDirect/{path}`

## Logging behavior
Running with:

```bash
python run.py
```

forces terminal DEBUG logging for app, API compatibility routes, backend manager events, and Comfy process stdout/stderr relay.

Preflight behavior:
- At startup, if `comfy.enable_preflight_checks` is enabled, the app runs a `--help` probe against the configured Comfy start script and checks for import/module errors.
- If preflight fails and `comfy.skip_auto_start_on_preflight_error` is true, auto-start is skipped and warnings are printed to terminal.

## Configuration
Primary config file:
- `/Users/jtreminio/www/python-swarmui/config.yaml`

Legacy fallback:
- `/Users/jtreminio/www/python-swarmui/config/settings.yaml`

The app prefers `config.yaml` when present. This file can be edited through `PUT /api/settings` or directly on disk.

Common settings:
- `server.host` and `server.port` for network binding
- `comfy.start_script` for your standalone ComfyUI entrypoint (for example: `/opt/comfyui/main.py`)
- `paths.model_roots` for model mounts
- `paths.output_root` for generated images
- `paths.wildcards_root` for wildcard files

## Detailed plan
See:
- `/Users/jtreminio/www/python-swarmui/PLAN.md`
