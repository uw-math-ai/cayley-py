from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .bfs import BfsLimitExceeded
from .challenge import forfeit_challenge, move_challenge, start_challenge
from .generators import list_families
from .graph_view import graph_summary, graph_view, shortest_paths_to_state
from .models import GraphSpec, GraphViewRequest


app = FastAPI(title="Cayley Graph Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _http_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/families")
def api_families() -> dict:
    return {"families": list_families()}


@app.post("/api/graph/summary")
def api_graph_summary(payload: dict) -> dict:
    try:
        return graph_summary(GraphSpec.from_dict(payload.get("spec", payload)))
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/graph/view")
def api_graph_view(payload: dict) -> dict:
    try:
        return graph_view(GraphViewRequest.from_dict(payload))
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/graph/shortest-paths")
def api_graph_shortest_paths(payload: dict) -> dict:
    try:
        target = payload.get("targetState", payload.get("target_state"))
        if target is None:
            raise ValueError("targetState is required")
        return shortest_paths_to_state(
            spec=GraphSpec.from_dict(payload.get("spec", {})),
            target=tuple(int(x) for x in target),
            layout=str(payload.get("layout", "layers")),
            exact_cap=int(payload.get("exactCap", payload.get("exact_cap", 500_000))),
            edge_cap=int(payload.get("edgeCap", payload.get("edge_cap", 80_000))),
        )
    except BfsLimitExceeded as exc:
        raise HTTPException(status_code=413, detail={"message": str(exc), "cap": exc.cap, "visited": exc.visited}) from exc
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/challenge/start")
def api_challenge_start(payload: dict) -> dict:
    try:
        session = start_challenge(
            spec=GraphSpec.from_dict(payload.get("spec", {})),
            difficulty=str(payload.get("difficulty", "medium")),
            custom_distance=payload.get("customDistance", payload.get("custom_distance")),
            seed=payload.get("seed"),
            exact_cap=int(payload.get("exactCap", payload.get("exact_cap", 500_000))),
        )
        return session.to_dict()
    except BfsLimitExceeded as exc:
        raise HTTPException(status_code=413, detail={"message": str(exc), "cap": exc.cap, "visited": exc.visited}) from exc
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/challenge/move")
def api_challenge_move(payload: dict) -> dict:
    try:
        return move_challenge(str(payload["sessionId"]), str(payload["generatorId"])).to_dict()
    except Exception as exc:
        raise _http_error(exc) from exc


@app.post("/api/challenge/forfeit")
def api_challenge_forfeit(payload: dict) -> dict:
    try:
        return forfeit_challenge(str(payload["sessionId"])).to_dict()
    except Exception as exc:
        raise _http_error(exc) from exc
