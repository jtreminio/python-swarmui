# Python SwarmUI Clone Plan

## Goal
Build a single-user Python application that is functionally similar to SwarmUI for core local usage:
- web UI for text-to-image generation
- ComfyUI backend self-start and monitoring
- model and wildcard path integration
- generation queue, job tracking, and output browsing

## Explicit Scope Constraints
- Ignore custom extensions under SwarmUI `src/Extensions`.
- Ignore user/role permissions; single local user has full access.
- Do not copy SwarmUI's `dlbackend/ComfyUI`; launch an existing ComfyUI install via `main.py` like SwarmUI does.

## Inputs Applied
Mount/default paths used by this clone:
- `swarmui_root`: `/Volumes/swarmui`
- `model_roots`: `/Volumes/models`
- `wildcards_root`: `/Volumes/aiconfigs/Wildcards`
- default comfy start script: `/Volumes/swarmui/dlbackend/ComfyUI/main.py`

## SwarmUI -> Python Mapping
1. SwarmUI `Program.cs` startup lifecycle:
- Python: app startup hook loads settings, ensures dirs, starts generation worker, optional backend autostart.

2. SwarmUI `Settings.cs`:
- Python: YAML config schema (`AppConfig`) with nested sections for paths, server, comfy launcher, generation defaults.

3. SwarmUI `NetworkBackendUtils.DoSelfStart`:
- Python: `ComfyProcessManager` that validates start script, allocates backend port, sets GPU env vars, resolves python executable, launches subprocess, monitors logs, and optional auto-restart.

4. SwarmUI `ComfyUISelfStartBackend.EnsureComfyFile`:
- Python: emit `data/comfy-auto-model.yaml` with model folder mappings and custom node forwarding for ComfyUI `--extra-model-paths-config`.

5. SwarmUI T2I flow and backend queueing:
- Python: single-worker queue service with submitted jobs, prompt expansion, workflow build, prompt submit to ComfyUI, history polling, and output persistence.

6. SwarmUI Web/API layer:
- Python: FastAPI routes for backend lifecycle, settings, models, wildcards, generation jobs, image serving, and Comfy direct proxy.

## Implementation Phases
1. Foundation
- project packaging (`pyproject.toml`), app package structure, config persistence, base FastAPI app.
- Acceptance: app starts and `/api/health` returns OK.

2. Backend Orchestration
- implement ComfyUI process manager matching Swarm startup behavior:
  - start script validation (`.py/.sh/.bat`)
  - venv/python executable detection
  - env vars (`CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`)
  - backend port selection from configured base
  - internal args (`--extra-model-paths-config`, preview mode, frontend version)
  - stdout/stderr tail capture
  - health-check wait and status transitions
  - stop/restart and optional auto-restart
- Acceptance: backend can be started/stopped from API and reaches healthy state.

3. Generation Core
- wildcard loader/expander from external wildcard mount.
- model scanner from configured roots.
- default txt2img workflow builder for Comfy API.
- async job queue and tracking with status transitions.
- output copy and metadata sidecar generation.
- Acceptance: generation endpoint returns job ID and completed jobs produce image outputs.

4. Web/API Surface
- endpoints:
  - health/status
  - backend lifecycle and logs
  - settings read/update
  - model list refresh
  - wildcard list
  - generate/job status/history
  - output image serving
  - Comfy direct HTTP proxy
- Acceptance: UI and API can drive an end-to-end generation.

5. Frontend Shell
- single-page interface with:
  - backend controls and status
  - generation form
  - model selector
  - job progress view
  - image gallery
- Acceptance: manual browser flow can run end-to-end.

6. Validation
- add unit tests for deterministic core logic (wildcards/workflow/model-yaml).
- run tests and sanity checks.
- Acceptance: tests pass locally.

## Risks and Mitigations
1. Comfy node/workflow compatibility varies by install.
- Mitigation: expose custom workflow passthrough; keep default workflow minimal and standard.

2. Comfy API progress variability across versions.
- Mitigation: reliable completion via history polling, optional progress enhancements later.

3. Environment-specific python/venv launch quirks.
- Mitigation: multiple executable resolution strategies and clear startup errors in API/UI.

## Deferred (Not in Initial Clone)
- Multi-user auth and roles.
- Extension framework parity with SwarmUI.
- Full Swarm parameter surface and dynamic node-driven parameter auto-generation.
- Electron/desktop launch modes.

## Clarifications Needed for Next Iteration
1. Should API route names mimic SwarmUI's `/API/...` reflection-based style for compatibility, or is a clean `/api/...` shape acceptable?
2. Do you want custom Comfy workflow upload/editing in this first version, or default txt2img workflow only?
3. Should outputs stay in this project (`data/output`) or be written directly into `/Volumes/swarmui/Output` for shared visibility?
