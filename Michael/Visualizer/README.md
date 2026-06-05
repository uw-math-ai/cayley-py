# Cayley Graph Visualizer

Local FastAPI + React/Vite app for exploring Cayley and Schreier graphs for `S_n`, plus an exact-BFS certified challenge mode.

## Features

- Generator catalog: adjacent/all/star transpositions, adjacent cycles, consecutive and wrapped `k`-cycles, pancake, Koltsov3, cubic pancake, and LRX.
- Graph spaces: full Cayley graphs and `k`-different coset Schreier graphs.
- Explorer views: full graph when small enough, sampled distance layers, local ego-neighborhoods, Bruhat rank, spectral, Lehmer projection, coset coordinates, and target-distance layout.
- Explorer interaction: visible edges, generator hover labels, node selection with dimmed background and certified shortest paths from identity.
- Challenge mode: compact setup controls, prominent goal card, side-by-side User/BFS graph canvases, a glowing goal marker on the User graph, edge-click moves, move chips, and shortest-path reveal on forfeit or non-optimal completion.
- Exact certification: independent BFS/predecessor cache under `Michael/Visualizer/.cache/bfs`.

There is no checkpoint inference path and no ML dependency set.

## Dependencies

Backend runtime dependencies are intentionally small:

```text
fastapi
uvicorn[standard]
```

The backend graph algorithms use the Python standard library. No `cayleypy`, NumPy, SciPy, PyTorch, scikit-learn, NetworkX, or checkpoint/model packages are required.

Frontend production dependencies are only:

```text
react
react-dom
```

Vite, TypeScript, Playwright, React types, and the Vite React plugin are dev dependencies because they are only needed for development, build, and tests.

## Run

Backend:

```bash
cd Michael/Visualizer/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd Michael/Visualizer/frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Challenge Mode

Challenge mode starts from identity and samples a target from an exact BFS distance layer. It only runs for graph specs that fit the exact BFS cap.

The User panel is the playable graph. Available moves are shown as move chips and highlighted outgoing edges from the current node; clicking a highlighted edge applies that generator. The target is also marked directly on the User graph.

The BFS panel is read-only. It advances one certified BFS step after each user move. If the user gives up or completes with a non-shortest path, the BFS panel and objective band reveal one certified shortest path.

## API

- `GET /api/families`
- `POST /api/graph/summary`
- `POST /api/graph/view`
- `POST /api/graph/shortest-paths`
- `POST /api/challenge/start`
- `POST /api/challenge/move`
- `POST /api/challenge/forfeit`

## Tests

Backend:

```bash
cd Michael/Visualizer/backend
.venv/bin/python -m unittest discover -s tests -v
```

Frontend:

```bash
cd Michael/Visualizer/frontend
npm run build
npm run test:e2e -- --reporter=list
```
