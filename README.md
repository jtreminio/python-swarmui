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
- SwarmUI root: `/Volumes/swarmui`
- Model root: `/Volumes/models`
- Wildcards root: `/Volumes/aiconfigs/Wildcards`
- Default Comfy start script: `/Volumes/swarmui/dlbackend/ComfyUI/main.py`

## Run
1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -e .
```

3. Start server:

```bash
uvicorn swarmui_clone.main:app --host 127.0.0.1 --port 7801 --reload
```

4. Open [http://127.0.0.1:7801/api/](http://127.0.0.1:7801/api/)

## API quick reference
- `GET /api/health`
- `GET /api/status`
- `GET /api/settings`
- `PUT /api/settings`
- `POST /api/backend/start`
- `POST /api/backend/stop`
- `POST /api/backend/restart`
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
- `GET|POST /API/ListLoadedModels`
- `GET|POST /API/SelectModel`
- `WS /API/SelectModelWS`
- `GET|POST /API/TriggerRefresh`
- `GET|POST /API/InterruptAll`
- `GET /View/{relative_path}`
- `GET /Output/{relative_path}`
- `ANY /ComfyBackendDirect/{path}`

## Configuration
Primary config file:
- `/Users/jtreminio/www/python-swarmui/config/settings.yaml`

It is auto-generated on first run and can be edited through `PUT /api/settings` or directly on disk.

## Detailed plan
See:
- `/Users/jtreminio/www/python-swarmui/PLAN.md`
