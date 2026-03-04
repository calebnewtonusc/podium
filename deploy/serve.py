"""
Podium FastAPI server — REST + WebSocket interface for competition runs.
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

app = FastAPI(title="Podium API", version="1.0.0")

# Allow callers to expand origins via CORS_ORIGINS env var (comma-separated).
# Defaults to localhost only so the wildcard is never used in production.
_cors_env = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class CompeteRequest(BaseModel):
    competition_url: str
    data_path: str
    time_limit_hours: float = 168.0


class CompeteResponse(BaseModel):
    session_id: str
    status: str
    message: str


@app.get("/health")
async def health():
    return {"status": "ok", "model": "podium-7b-v1"}


@app.post("/compete", response_model=CompeteResponse)
async def start_competition(req: CompeteRequest):
    """Start a new competition session. Returns session_id for status tracking."""
    session_id = str(uuid.uuid4())[:8]
    output_dir = f"./results/{session_id}"

    # Launch competition runner in background
    asyncio.create_task(_run_competition(session_id, req, output_dir))

    return CompeteResponse(
        session_id=session_id,
        status="started",
        message=f"Competition started. Track progress via WebSocket: /ws/{session_id}",
    )


@app.get("/compete/{session_id}")
async def get_competition_status(session_id: str):
    """Get current status and results for a competition session."""
    results_path = Path(f"./results/{session_id}/results.json")
    if results_path.exists():
        with open(results_path) as f:
            return {"status": "complete", "results": json.load(f)}
    return {"status": "running", "session_id": session_id}


@app.websocket("/ws/{session_id}")
async def competition_stream(websocket: WebSocket, session_id: str):
    """Stream real-time progress updates for a competition session."""
    await websocket.accept()
    try:
        # Poll results file for updates and stream to client
        while True:
            results_path = Path(f"./results/{session_id}/results.json")
            if results_path.exists():
                try:
                    with open(results_path) as f:
                        results = json.load(f)
                    await websocket.send_json({"type": "complete", "results": results})
                    break
                except json.JSONDecodeError:
                    pass  # File still being written by background runner — keep polling

            log_path = Path(f"./results/{session_id}/progress.log")
            if log_path.exists():
                text = log_path.read_text().strip()
                last_line = text.split("\n")[-1] if text else "working..."
                await websocket.send_json({"type": "progress", "message": last_line})

            # TODO (PO-63): Replace polling with an event-driven mechanism (e.g. asyncio.Event
            # set by _run_competition) to avoid the fixed 5-second latency on completion.
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        pass


async def _run_competition(session_id: str, req: CompeteRequest, output_dir: str):
    """Background task: run full competition pipeline."""
    import os
    from agents.competition_runner import CompetitionRunner

    model_path = os.environ.get("MODEL_PATH", "./checkpoints/dpo")
    runner = CompetitionRunner(model_path=model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        await runner.compete(
            competition_url=req.competition_url,
            data_path=req.data_path,
            output_dir=output_dir,
            time_budget_hours=req.time_limit_hours,
        )
    except Exception as e:
        logger.error(f"Competition {session_id} failed: {e}")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{output_dir}/results.json", "w") as f:
            json.dump({"error": str(e), "status": "failed"}, f)
