from __future__ import annotations

import uvicorn


if __name__ == "__main__":
    uvicorn.run("swarmui_clone.main:app", host="127.0.0.1", port=7801, reload=True)
