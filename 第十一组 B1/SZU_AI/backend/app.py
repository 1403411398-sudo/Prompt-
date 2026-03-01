"""
Main FastAPI Application - Prompt Optimization System
Serves the web frontend and provides API endpoints for optimization.
"""
import json
import os
import sys
import time
import logging
import threading
import asyncio
import queue
from pathlib import Path
from typing import Optional
from collections import Counter

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.api_client import call_llm, call_llm_batch, evaluate_prompts_parallel, get_available_models
from backend.prompt_generator import (
    generate_prompts_for_task,
    ROLE_KEYWORDS, STYLE_KEYWORDS_CLS, CONSTRAINT_KEYWORDS,
    COT_KEYWORDS, SUMMARY_LENGTH_KEYWORDS, TRANSLATION_STYLE_KEYWORDS,
)
from backend.evaluator import evaluate
from backend.optimizers.base import OptimizationResult
from backend.optimizers.random_search import RandomSearchOptimizer
from backend.optimizers.genetic import GeneticOptimizer
from backend.optimizers.bayesian import BayesianOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Prompt Optimization System", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
DATA_DIR = Path(__file__).parent.parent / "data"

# Global state for active optimization sessions
active_sessions = {}
ws_connections: dict[str, list[WebSocket]] = {}

ALL_KEYWORDS_LIST = (ROLE_KEYWORDS + STYLE_KEYWORDS_CLS + CONSTRAINT_KEYWORDS +
                     COT_KEYWORDS + SUMMARY_LENGTH_KEYWORDS + TRANSLATION_STYLE_KEYWORDS)
ALL_KEYWORDS_LIST = [kw for kw in ALL_KEYWORDS_LIST if kw]


# ──────────────────────── Data Loading ────────────────────────

def load_data(task_type: str) -> list[dict]:
    """Load dataset for a given task type."""
    filepath = DATA_DIR / f"{task_type}_data.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────── Keyword Contribution Analysis ────────────────────────

def analyze_keyword_contributions(results: list[dict]) -> dict:
    """
    Analyze the contribution of each keyword to prompt performance.
    Returns a dict mapping keyword -> average score delta.
    """
    keyword_scores = {kw: [] for kw in ALL_KEYWORDS_LIST}
    keyword_absent_scores = {kw: [] for kw in ALL_KEYWORDS_LIST}

    for r in results:
        prompt = r.get("prompt", "")
        score = r.get("score", 0.0)
        for kw in ALL_KEYWORDS_LIST:
            if kw in prompt:
                keyword_scores[kw].append(score)
            else:
                keyword_absent_scores[kw].append(score)

    contributions = {}
    for kw in ALL_KEYWORDS_LIST:
        present = keyword_scores[kw]
        absent = keyword_absent_scores[kw]
        if present and absent:
            avg_present = sum(present) / len(present)
            avg_absent = sum(absent) / len(absent)
            contributions[kw] = round(avg_present - avg_absent, 4)
        elif present:
            contributions[kw] = round(sum(present) / len(present), 4)
        else:
            contributions[kw] = 0.0

    # Sort by contribution
    contributions = dict(sorted(contributions.items(), key=lambda x: x[1], reverse=True))
    return contributions


# ──────────────────────── Request/Response Models ────────────────────────

class OptimizeRequest(BaseModel):
    task_type: str  # classification, summarization, translation
    algorithm: str  # random_search, genetic, bayesian
    max_iterations: int = 10
    use_llm_judge: bool = False
    model: str = "qwen3.5-35b-a3b"
    custom_prompts: Optional[list[str]] = None
    data_indices: Optional[list[int]] = None  # Selected data sample indices


class TestPromptRequest(BaseModel):
    task_type: str
    prompt: str
    model: str = "qwen3.5-35b-a3b"


class MultiTaskRequest(BaseModel):
    task_types: list[str]
    algorithm: str
    max_iterations: int = 8
    use_llm_judge: bool = False
    model: str = "qwen3.5-35b-a3b"
    data_indices: Optional[dict[str, list[int]]] = None  # Per-task data indices


# ──────────────────────── API Endpoints ────────────────────────

@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "time": time.time()}


@app.get("/api/models")
async def list_models():
    """List available LLM models."""
    return {"models": get_available_models()}


@app.get("/api/tasks")
async def list_tasks():
    """List available tasks and their data."""
    tasks = {}
    for task_type in ["classification", "summarization", "translation"]:
        try:
            data = load_data(task_type)
            tasks[task_type] = {
                "name": {
                    "classification": "文本分类",
                    "summarization": "文本摘要",
                    "translation": "中英翻译",
                }[task_type],
                "count": len(data),
                "sample": data[0] if data else {},
            }
        except FileNotFoundError:
            pass
    return tasks


@app.get("/api/task_data/{task_type}")
async def get_task_data(task_type: str):
    """Return full dataset for a task type with indices."""
    try:
        data = load_data(task_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    items = []
    for i, d in enumerate(data):
        item = {"index": i}
        if task_type == "classification":
            item["text"] = d["text"]
            item["label"] = d["label"]
        elif task_type == "summarization":
            item["text"] = d["text"][:120] + "..." if len(d["text"]) > 120 else d["text"]
            item["reference"] = d["reference"][:80] + "..." if len(d["reference"]) > 80 else d["reference"]
        elif task_type == "translation":
            item["source"] = d["source"]
            item["reference"] = d["reference"][:80] + "..." if len(d["reference"]) > 80 else d["reference"]
        items.append(item)

    return {"task_type": task_type, "count": len(items), "data": items}


@app.get("/api/prompts/{task_type}")
async def get_prompts(task_type: str, n: int = 10):
    """Generate prompt candidates for a task."""
    prompts = generate_prompts_for_task(task_type, n)
    return {"task_type": task_type, "prompts": prompts}


@app.post("/api/test_prompt")
async def test_prompt(req: TestPromptRequest):
    """Test a single prompt on the dataset."""
    try:
        data = load_data(req.task_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if req.task_type == "classification":
        inputs = [d["text"] for d in data]
        references = [d["label"] for d in data]
    elif req.task_type == "summarization":
        inputs = [d["text"] for d in data]
        references = [d["reference"] for d in data]
    elif req.task_type == "translation":
        inputs = [d["source"] for d in data]
        references = [d["reference"] for d in data]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {req.task_type}")

    predictions = call_llm_batch(req.prompt, inputs, model=req.model)
    metrics = evaluate(req.task_type, predictions, references, inputs)

    return {
        "prompt": req.prompt,
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "inputs": inputs,
    }


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    """Start a prompt optimization run."""
    session_id = f"{req.task_type}_{req.algorithm}_{int(time.time())}"

    try:
        data = load_data(req.task_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Filter by selected indices
    if req.data_indices is not None and len(req.data_indices) > 0:
        data = [data[i] for i in req.data_indices if i < len(data)]

    # Prepare data
    if req.task_type == "classification":
        inputs = [d["text"] for d in data]
        references = [d["label"] for d in data]
    elif req.task_type == "summarization":
        inputs = [d["text"] for d in data]
        references = [d["reference"] for d in data]
    elif req.task_type == "translation":
        inputs = [d["source"] for d in data]
        references = [d["reference"] for d in data]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {req.task_type}")

    # Create evaluation function
    def evaluate_fn(prompt: str) -> tuple[float, dict, list[str]]:
        predictions = call_llm_batch(prompt, inputs, model=req.model)
        metrics = evaluate(req.task_type, predictions, references, inputs, req.use_llm_judge)
        return metrics["primary_score"], metrics, predictions

    # Generate initial prompts
    if req.custom_prompts:
        initial_prompts = req.custom_prompts
    else:
        initial_prompts = generate_prompts_for_task(req.task_type, max(10, req.max_iterations))

    # Select optimizer
    if req.algorithm == "random_search":
        optimizer = RandomSearchOptimizer(req.task_type, evaluate_fn, req.max_iterations)
    elif req.algorithm == "genetic":
        optimizer = GeneticOptimizer(req.task_type, evaluate_fn, req.max_iterations)
    elif req.algorithm == "bayesian":
        optimizer = BayesianOptimizer(req.task_type, evaluate_fn, req.max_iterations)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {req.algorithm}")

    # Results collector for real-time updates
    results_collector = []

    def on_result(result: OptimizationResult):
        result_dict = {
            "prompt": result.prompt,
            "score": result.score,
            "metrics": result.metrics,
            "iteration": result.iteration,
        }
        results_collector.append(result_dict)
        # Send to WebSocket clients
        _broadcast_to_session(session_id, {
            "type": "iteration_result",
            "session_id": session_id,
            "data": result_dict,
        })

    optimizer.set_callback(on_result)

    # Store session info
    active_sessions[session_id] = {
        "status": "running",
        "task_type": req.task_type,
        "algorithm": req.algorithm,
        "started_at": time.time(),
    }

    # Run optimization (synchronous for now)
    try:
        history = optimizer.optimize(initial_prompts)

        # Compute keyword contributions
        keyword_contrib = analyze_keyword_contributions(
            [{"prompt": r.prompt, "score": r.score} for r in history.results]
        )
        history.keyword_contributions = keyword_contrib

        result = history.to_dict()
        active_sessions[session_id]["status"] = "completed"
        active_sessions[session_id]["result"] = result

        return {"session_id": session_id, "result": result}

    except Exception as e:
        active_sessions[session_id]["status"] = "failed"
        active_sessions[session_id]["error"] = str(e)
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/optimize_stream")
async def optimize_stream(req: OptimizeRequest):
    """Start a prompt optimization run with SSE streaming results."""
    try:
        data = load_data(req.task_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Filter by selected indices
    if req.data_indices is not None and len(req.data_indices) > 0:
        data = [data[i] for i in req.data_indices if i < len(data)]

    # Prepare data
    if req.task_type == "classification":
        inputs = [d["text"] for d in data]
        references = [d["label"] for d in data]
    elif req.task_type == "summarization":
        inputs = [d["text"] for d in data]
        references = [d["reference"] for d in data]
    elif req.task_type == "translation":
        inputs = [d["source"] for d in data]
        references = [d["reference"] for d in data]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {req.task_type}")

    # Create evaluation function
    def evaluate_fn(prompt: str) -> tuple[float, dict, list[str]]:
        predictions = call_llm_batch(prompt, inputs, model=req.model)
        metrics = evaluate(req.task_type, predictions, references, inputs, req.use_llm_judge)
        return metrics["primary_score"], metrics, predictions

    # Generate initial prompts
    if req.custom_prompts:
        initial_prompts = req.custom_prompts
    else:
        initial_prompts = generate_prompts_for_task(req.task_type, max(10, req.max_iterations))

    # Select optimizer
    if req.algorithm == "random_search":
        optimizer = RandomSearchOptimizer(req.task_type, evaluate_fn, req.max_iterations)
    elif req.algorithm == "genetic":
        optimizer = GeneticOptimizer(req.task_type, evaluate_fn, req.max_iterations)
    elif req.algorithm == "bayesian":
        optimizer = BayesianOptimizer(req.task_type, evaluate_fn, req.max_iterations)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown algorithm: {req.algorithm}")

    # Thread-safe queue for SSE
    result_queue = queue.Queue()

    def on_result(result: OptimizationResult):
        result_queue.put({
            "type": "iteration",
            "prompt": result.prompt,
            "score": result.score,
            "metrics": result.metrics,
            "iteration": result.iteration,
        })

    optimizer.set_callback(on_result)

    def run_optimization():
        try:
            history = optimizer.optimize(initial_prompts)
            keyword_contrib = analyze_keyword_contributions(
                [{"prompt": r.prompt, "score": r.score} for r in history.results]
            )
            history.keyword_contributions = keyword_contrib
            result_queue.put({"type": "complete", "result": history.to_dict()})
        except Exception as e:
            logger.error(f"Streaming optimization failed: {e}", exc_info=True)
            result_queue.put({"type": "error", "message": str(e)})

    async def event_generator():
        thread = threading.Thread(target=run_optimization, daemon=True)
        thread.start()

        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(None, lambda: result_queue.get(timeout=120))
                yield f"data: {json.dumps(msg, ensure_ascii=False)}\n\n"
                if msg["type"] in ("complete", "error"):
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/optimize_multi")
async def optimize_multi(req: MultiTaskRequest):
    """Run optimization across multiple tasks."""
    results = {}

    for task_type in req.task_types:
        try:
            single_req = OptimizeRequest(
                task_type=task_type,
                algorithm=req.algorithm,
                max_iterations=req.max_iterations,
                use_llm_judge=req.use_llm_judge,
                model=req.model,
            )
            response = await optimize(single_req)
            results[task_type] = response
        except Exception as e:
            results[task_type] = {"error": str(e)}

    return {"results": results}


@app.get("/api/sessions")
async def list_sessions():
    """List all optimization sessions."""
    return {
        sid: {
            "status": info["status"],
            "task_type": info["task_type"],
            "algorithm": info["algorithm"],
            "started_at": info["started_at"],
        }
        for sid, info in active_sessions.items()
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details."""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return active_sessions[session_id]


# ──────────────────────── WebSocket ────────────────────────

def _broadcast_to_session(session_id: str, message: dict):
    """Broadcast message to all WebSocket clients for a session."""
    # This is simplified - in production use async properly
    pass


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    if session_id not in ws_connections:
        ws_connections[session_id] = []
    ws_connections[session_id].append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_connections[session_id].remove(websocket)


# ──────────────────────── Serve Frontend Static Files ────────────────────────

# Mount static files last so API routes take priority
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
