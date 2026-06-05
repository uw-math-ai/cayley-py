import { useEffect, useMemo, useState } from "react";
import GraphCanvas from "./GraphCanvas";
import {
  fetchFamilies,
  fetchShortestPaths,
  fetchSummary,
  fetchView,
  forfeitChallenge,
  moveChallenge,
  startChallenge
} from "./api";
import type { ChallengeState, FamilyDef, GraphEdge, GraphNode, GraphSpec, GraphView, ShortestPaths, Summary } from "./types";

const DEFAULT_SPEC: GraphSpec = {
  family: "koltsov3",
  n: 5,
  params: { permType: 1, k: 0, d: 2, stateSpace: "cayley", differentK: 2 },
  inversePolicy: "default"
};

function formatPerm(state?: number[] | null) {
  if (!state) return "-";
  const shown = state.map((x) => x + 1);
  return shown.length <= 9 ? shown.join("") : `[${shown.join(" ")}]`;
}

function defaultsFor(family: FamilyDef | undefined): Record<string, number | string> {
  const params: Record<string, number | string> = {};
  family?.parameters.forEach((param) => {
    params[param.id] = param.default;
  });
  return params;
}

function scopeParams(params: GraphSpec["params"]): Record<string, number | string> {
  return {
    stateSpace: params.stateSpace ?? "cayley",
    differentK: params.differentK ?? 2
  };
}

function stateId(state?: number[] | null) {
  return state ? state.join(",") : "";
}

function edgeKey(edge: GraphEdge) {
  return `${edge.source}|${edge.target}|${edge.generatorId}`;
}

function applyGenerator(state: number[], generator: number[]) {
  return generator.map((index) => state[index]);
}

function uniqueStates(states: number[][]): number[][] {
  const seen = new Set<string>();
  const out: number[][] = [];
  states.forEach((state) => {
    const id = stateId(state);
    if (!id || seen.has(id)) return;
    seen.add(id);
    out.push(state);
  });
  return out;
}

function pinnedStatesForChallenge(challenge: ChallengeState): number[][] {
  const currentTargets =
    challenge.status === "active"
      ? challenge.generators.map((generator) => applyGenerator(challenge.current, generator.permutation))
      : [];
  return uniqueStates([
    ...(challenge.userStates ?? []),
    ...(challenge.certifiedPath?.states ?? []),
    ...currentTargets,
    challenge.current,
    challenge.target
  ]);
}

function buildPathOverlay(
  states: number[][],
  generatorIds: string[],
  nodesById: Map<string, GraphNode>,
  generators: ChallengeState["generators"],
  focusEdgeIndex = -1
) {
  if (!states.length) return null;
  const generatorLabels = new Map(generators.map((gen) => [gen.id, gen.label]));
  const edgeCount = Math.min(generatorIds.length, Math.max(0, states.length - 1));
  const edges = generatorIds.slice(0, edgeCount).map((generatorId, index) => ({
    source: stateId(states[index]),
    target: stateId(states[index + 1]),
    generatorId,
    generatorLabel: generatorLabels.get(generatorId) ?? generatorId,
    forwardLayer: true
  }));
  const nodeIds = uniqueStates(states).map(stateId);
  const edgeKeys = edges.map(edgeKey);
  const focusEdgeKeys = focusEdgeIndex >= 0 && focusEdgeIndex < edgeKeys.length ? [edgeKeys[focusEdgeIndex]] : [];
  return {
    targetId: stateId(states[states.length - 1]),
    nodeIds,
    edgeKeys,
    focusEdgeKeys,
    nodes: nodeIds.map((id) => nodesById.get(id)).filter((node): node is GraphNode => Boolean(node)),
    edges
  };
}

export default function App() {
  const [families, setFamilies] = useState<FamilyDef[]>([]);
  const [activeTab, setActiveTab] = useState<"explorer" | "challenge">("explorer");
  const [spec, setSpec] = useState<GraphSpec>(DEFAULT_SPEC);
  const [mode, setMode] = useState("auto");
  const [layout, setLayout] = useState("layers");
  const [challengeLayout, setChallengeLayout] = useState("layers");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [view, setView] = useState<GraphView | null>(null);
  const [viewSpec, setViewSpec] = useState<GraphSpec | null>(null);
  const [selectedPaths, setSelectedPaths] = useState<ShortestPaths | null>(null);
  const [challenge, setChallenge] = useState<ChallengeState | null>(null);
  const [challengeView, setChallengeView] = useState<GraphView | null>(null);
  const [difficulty, setDifficulty] = useState("medium");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const family = useMemo(() => families.find((item) => item.id === spec.family), [families, spec.family]);
  const stateSpace = String(spec.params.stateSpace ?? "cayley");
  const viewNodesById = useMemo(() => new Map((view?.nodes ?? []).map((node) => [node.id, node])), [view]);
  const challengeNodesById = useMemo(() => new Map((challengeView?.nodes ?? []).map((node) => [node.id, node])), [challengeView]);
  const pathOverlay = useMemo(() => {
    if (!selectedPaths) return null;
    return {
      ...selectedPaths,
      nodes: selectedPaths.nodes.map((node) => viewNodesById.get(node.id) ?? node)
    };
  }, [selectedPaths, viewNodesById]);
  const userTrailOverlay = useMemo(() => {
    if (!challenge) return null;
    const states = challenge.userStates?.length ? challenge.userStates : [challenge.start];
    return buildPathOverlay(states, challenge.userPath, challengeNodesById, challenge.generators, Math.max(0, challenge.userPath.length - 1));
  }, [challenge, challengeNodesById]);
  const bfsShowFullPath = Boolean(challenge && (challenge.status === "forfeited" || challenge.status === "completed"));
  const bfsStepCount = useMemo(() => {
    if (!challenge?.certifiedPath) return 0;
    return bfsShowFullPath ? challenge.certifiedPath.length : Math.min(challenge.userPath.length, challenge.certifiedPath.length);
  }, [challenge, bfsShowFullPath]);
  const bfsVisibleStates = useMemo(() => challenge?.certifiedPath?.states.slice(0, bfsStepCount + 1) ?? [], [challenge, bfsStepCount]);
  const bfsVisibleMoves = useMemo(() => challenge?.certifiedPath?.generatorIds.slice(0, bfsStepCount) ?? [], [challenge, bfsStepCount]);
  const bfsTrailOverlay = useMemo(() => {
    if (!challenge?.certifiedPath) return null;
    return buildPathOverlay(bfsVisibleStates, bfsVisibleMoves, challengeNodesById, challenge.generators, Math.max(0, bfsVisibleMoves.length - 1));
  }, [challenge, bfsVisibleMoves, bfsVisibleStates, challengeNodesById]);
  const currentStateId = stateId(challenge?.current);
  const bfsCurrentStateId = stateId(bfsVisibleStates[bfsVisibleStates.length - 1] ?? challenge?.start);
  const legalChallengeEdges = useMemo(
    () => (challenge?.status === "active" ? (challengeView?.edges ?? []).filter((edge) => edge.source === currentStateId) : []),
    [challenge?.status, challengeView, currentStateId]
  );
  const legalChallengeEdgeKeys = useMemo(() => legalChallengeEdges.map(edgeKey), [legalChallengeEdges]);
  const legalMoveLabels = useMemo(() => {
    const seen = new Set<string>();
    return legalChallengeEdges
      .map((edge) => edge.generatorLabel || edge.generatorId)
      .filter((label) => {
        if (seen.has(label)) return false;
        seen.add(label);
        return true;
      });
  }, [legalChallengeEdges]);
  const revealFullBfsText = Boolean(
    challenge &&
      (challenge.status === "forfeited" ||
        (challenge.status === "completed" && (challenge.excess ?? 0) > 0))
  );
  const shownBfsMove = (() => {
    if (!challenge?.certifiedPath || challenge.userPath.length === 0) return "id";
    if (challenge.userPath.length > challenge.certifiedPath.generatorIds.length) return "done";
    return challenge.certifiedPath.generatorIds[challenge.userPath.length - 1] ?? "done";
  })();
  const lastUserMove = challenge?.userPath[challenge.userPath.length - 1] ?? "id";
  const userWord = challenge?.userPath.length ? challenge.userPath.join(" ") : "id";
  const bfsWord = bfsVisibleMoves.length ? bfsVisibleMoves.join(" ") : "id";

  const clearSessionState = () => {
    setChallenge(null);
    setChallengeView(null);
    setSelectedPaths(null);
  };

  useEffect(() => {
    fetchFamilies()
      .then((items) => {
        setFamilies(items);
        const initial = items.find((item) => item.id === DEFAULT_SPEC.family);
        if (initial) setSpec((current) => ({ ...current, params: { ...defaultsFor(initial), ...scopeParams(current.params) } }));
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  const load = async (nextMode = mode, nextLayout = layout, nextSpec = spec) => {
    setBusy(true);
    setError("");
    try {
      const [nextSummary, nextView] = await Promise.all([fetchSummary(nextSpec), fetchView(nextSpec, nextMode, { layout: nextLayout })]);
      setSummary(nextSummary);
      setView(nextView);
      setViewSpec(nextSpec);
      setSelectedPaths(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (families.length) void load("auto");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [families.length]);

  const updateFamily = (familyId: string) => {
    const nextFamily = families.find((item) => item.id === familyId);
    clearSessionState();
    setSpec({
      family: familyId,
      n: Math.max(nextFamily?.minN ?? 2, spec.n),
      params: { ...defaultsFor(nextFamily), ...scopeParams(spec.params) },
      inversePolicy: "default"
    });
  };

  const updateParam = (id: string, value: string) => {
    const numeric = Number(value);
    clearSessionState();
    setSpec((current) => ({
      ...current,
      params: { ...current.params, [id]: Number.isFinite(numeric) ? numeric : value }
    }));
  };

  const updateStateSpace = (nextSpace: string) => {
    clearSessionState();
    setSpec((current) => ({
      ...current,
      params: { ...current.params, stateSpace: nextSpace, differentK: current.params.differentK ?? 2 }
    }));
  };

  const loadChallengeGraph = (next: ChallengeState, nextLayout = challengeLayout) =>
    fetchView(next.spec, "layers", {
      layout: nextLayout,
      targetState: next.target,
      pinnedStates: pinnedStatesForChallenge(next),
      radius: 2
    });

  const onStartChallenge = async () => {
    setBusy(true);
    setError("");
    try {
      const nextSummary = await fetchSummary(spec);
      setSummary(nextSummary);
      const next = await startChallenge(spec, difficulty);
      setChallenge(next);
      setChallengeView(await loadChallengeGraph(next));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onMove = async (generatorId: string) => {
    if (!challenge) return;
    setBusy(true);
    setError("");
    try {
      const next = await moveChallenge(challenge.sessionId, generatorId);
      setChallenge(next);
      setChallengeView(await loadChallengeGraph(next));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onChallengeEdgeClick = (edge: GraphEdge) => {
    if (!legalChallengeEdgeKeys.includes(edgeKey(edge))) return;
    void onMove(edge.generatorId);
  };

  const onForfeit = async () => {
    if (!challenge) return;
    setBusy(true);
    try {
      const next = await forfeitChallenge(challenge.sessionId);
      setChallenge(next);
      setChallengeView(await loadChallengeGraph(next));
    } finally {
      setBusy(false);
    }
  };

  const onSelectNode = async (node: GraphNode | null) => {
    if (!node) {
      setSelectedPaths(null);
      return;
    }
    if (!viewSpec) return;
    setBusy(true);
    setError("");
    try {
      setSelectedPaths(await fetchShortestPaths(viewSpec, node.state, layout));
    } catch (err) {
      setSelectedPaths(null);
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="app-frame">
      <header className="topbar">
        <div className="brand">
          <span className="mark">C</span>
          <div>
            <h1>Cayley</h1>
            <p>S_n visualizer</p>
          </div>
        </div>
        <div className="tabbar" role="tablist">
          <button role="tab" aria-selected={activeTab === "explorer"} className={activeTab === "explorer" ? "tab-button active" : "tab-button"} onClick={() => setActiveTab("explorer")}>
            Explorer
          </button>
          <button role="tab" aria-selected={activeTab === "challenge"} className={activeTab === "challenge" ? "tab-button active" : "tab-button"} onClick={() => setActiveTab("challenge")}>
            Challenge
          </button>
        </div>
      </header>

      {activeTab === "explorer" ? (
        <section className="tab-shell explorer-shell" role="tabpanel" aria-label="Explorer">
          <aside className="panel explorer-panel">
            <section className="controls">
              <label>
                Family
                <select value={spec.family} onChange={(event) => updateFamily(event.target.value)}>
                  {families.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <div className="inline">
                <label>
                  n
                  <input
                    type="number"
                    min={family?.minN ?? 2}
                    value={spec.n}
                    onChange={(event) => {
                      clearSessionState();
                      setSpec((current) => ({ ...current, n: Number(event.target.value) }));
                    }}
                  />
                </label>
                <label>
                  Graph
                  <select value={stateSpace} onChange={(event) => updateStateSpace(event.target.value)}>
                    <option value="cayley">Cayley</option>
                    <option value="k_different">k-different</option>
                  </select>
                </label>
              </div>
              {stateSpace === "k_different" && (
                <label>
                  different k
                  <input type="number" min={2} max={spec.n} value={String(spec.params.differentK ?? 2)} onChange={(event) => updateParam("differentK", event.target.value)} />
                </label>
              )}
              {family?.parameters.map((param) => (
                <label key={param.id}>
                  {param.label}
                  {param.type === "select" ? (
                    <select value={String(spec.params[param.id] ?? param.default)} onChange={(event) => updateParam(param.id, event.target.value)}>
                      {param.options?.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input type="number" min={param.min ?? 0} value={String(spec.params[param.id] ?? param.default)} onChange={(event) => updateParam(param.id, event.target.value)} />
                  )}
                </label>
              ))}
              <label>
                Inverses
                <select
                  value={spec.inversePolicy}
                  onChange={(event) => {
                    clearSessionState();
                    setSpec((current) => ({ ...current, inversePolicy: event.target.value as GraphSpec["inversePolicy"] }));
                  }}
                >
                  <option value="default">default</option>
                  <option value="listed">listed</option>
                  <option value="closed">closed</option>
                </select>
              </label>
              <button className="wide primary-view" onClick={() => void load(mode, layout)} disabled={busy}>
                View
              </button>
            </section>

            <section className="explorer-actions">
              <div className="section-head">
                <h2>Explorer</h2>
                <span className="panel-badge">{summary?.specHash?.slice(0, 8) ?? "hash"}</span>
              </div>
              <div className="inline">
                <label>
                  View
                  <select
                    value={mode}
                    onChange={(event) => {
                      setMode(event.target.value);
                      void load(event.target.value, layout);
                    }}
                  >
                    <option value="auto">auto</option>
                    <option value="full">full</option>
                    <option value="layers">layers</option>
                    <option value="local">local</option>
                  </select>
                </label>
                <label>
                  Layout
                  <select
                    value={layout}
                    onChange={(event) => {
                      setLayout(event.target.value);
                      setSelectedPaths(null);
                      void load(mode, event.target.value);
                    }}
                  >
                    <option value="layers">layers</option>
                    <option value="bruhat">bruhat</option>
                    <option value="spectral">spectral</option>
                    <option value="lehmer">lehmer</option>
                    <option value="coset">coset</option>
                    <option value="target-distance">target</option>
                  </select>
                </label>
              </div>
              <section className="stats">
                <span>{summary?.generatorCount ?? 0} gens</span>
                <span>{summary?.exact?.diameter != null ? `diam ${summary.exact.diameter}` : "no cert"}</span>
                <span>{summary?.stateSpace ?? stateSpace}</span>
                <span>{view?.layout ?? layout}</span>
              </section>

              {selectedPaths && (
                <div className="path-readout selected-path" data-testid="selected-path">
                  <span>Selected</span>
                  <strong>{formatPerm(selectedPaths.target)}</strong>
                  <span>Length</span>
                  <strong>{selectedPaths.length}</strong>
                  <span>Paths</span>
                  <strong>
                    {selectedPaths.pathCount.toLocaleString()}
                    {selectedPaths.pathCountCapped ? "+" : ""}
                  </strong>
                  <span>Word</span>
                  <strong>{selectedPaths.canonicalPath.generatorIds.join(" ") || "id"}</strong>
                  {selectedPaths.truncated && (
                    <>
                      <span>Status</span>
                      <strong>sampled</strong>
                    </>
                  )}
                </div>
              )}
            </section>
          </aside>

          <section className="stage explorer-stage">
            <GraphCanvas view={view} testId="explorer-canvas" selectedNodeId={selectedPaths?.targetId} pathOverlay={pathOverlay} onNodeClick={onSelectNode} />
            {error && <div className="error-strip">{error}</div>}
            {busy && <div className="busy-strip">Working</div>}
          </section>
        </section>
      ) : (
        <section className="tab-shell challenge-shell" role="tabpanel" aria-label="Challenge">
          <section className="challenge-setup">
            <div className="challenge-setup-grid">
              <label>
                Family
                <select value={spec.family} onChange={(event) => updateFamily(event.target.value)}>
                  {families.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                n
                <input
                  type="number"
                  min={family?.minN ?? 2}
                  value={spec.n}
                  onChange={(event) => {
                    clearSessionState();
                    setSpec((current) => ({ ...current, n: Number(event.target.value) }));
                  }}
                />
              </label>
              <label>
                Graph
                <select value={stateSpace} onChange={(event) => updateStateSpace(event.target.value)}>
                  <option value="cayley">Cayley</option>
                  <option value="k_different">k-different</option>
                </select>
              </label>
              {stateSpace === "k_different" && (
                <label>
                  different k
                  <input type="number" min={2} max={spec.n} value={String(spec.params.differentK ?? 2)} onChange={(event) => updateParam("differentK", event.target.value)} />
                </label>
              )}
              {family?.parameters.map((param) => (
                <label key={param.id}>
                  {param.label}
                  {param.type === "select" ? (
                    <select value={String(spec.params[param.id] ?? param.default)} onChange={(event) => updateParam(param.id, event.target.value)}>
                      {param.options?.map((option) => (
                        <option key={option} value={option}>
                          {option}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input type="number" min={param.min ?? 0} value={String(spec.params[param.id] ?? param.default)} onChange={(event) => updateParam(param.id, event.target.value)} />
                  )}
                </label>
              ))}
              <label>
                Inverses
                <select
                  value={spec.inversePolicy}
                  onChange={(event) => {
                    clearSessionState();
                    setSpec((current) => ({ ...current, inversePolicy: event.target.value as GraphSpec["inversePolicy"] }));
                  }}
                >
                  <option value="default">default</option>
                  <option value="listed">listed</option>
                  <option value="closed">closed</option>
                </select>
              </label>
              <label>
                Layout
                <select
                  value={challengeLayout}
                  onChange={(event) => {
                    const nextLayout = event.target.value;
                    setChallengeLayout(nextLayout);
                    if (challenge) void loadChallengeGraph(challenge, nextLayout).then(setChallengeView);
                  }}
                >
                  <option value="target-distance">target</option>
                  <option value="layers">layers</option>
                  <option value="spectral">spectral</option>
                  <option value="lehmer">lehmer</option>
                  <option value="coset">coset</option>
                  <option value="bruhat">bruhat</option>
                </select>
              </label>
              <label>
                Level
                <select value={difficulty} onChange={(event) => setDifficulty(event.target.value)}>
                  <option value="easy">easy</option>
                  <option value="medium">medium</option>
                  <option value="hard">hard</option>
                </select>
              </label>
              <button className="wide setup-button" onClick={() => void onStartChallenge()} disabled={busy}>
                New
              </button>
              {challenge?.status === "active" && (
                <button className="setup-button danger-button" onClick={() => void onForfeit()} disabled={busy}>
                  Give up
                </button>
              )}
            </div>
            {challenge && (
              <div className="challenge-objective" data-testid="challenge-objective">
                <div className="goal-card">
                  <span>Goal</span>
                  <strong>{formatPerm(challenge.target)}</strong>
                  <small>Start {formatPerm(challenge.start)}</small>
                </div>
                <div className="challenge-state-cards">
                  <div>
                    <span>User</span>
                    <strong data-testid="objective-user-word">{userWord}</strong>
                  </div>
                  <div>
                    <span>BFS</span>
                    <strong data-testid="objective-bfs-word">{bfsWord}</strong>
                  </div>
                  <div className="score-card" data-testid="score-card">
                    <span>Score</span>
                    <strong>{challenge.userLength}/{challenge.optimalLength ?? challenge.targetDistance}</strong>
                  </div>
                  <div>
                    <span>Status</span>
                    <strong>{challenge.status === "completed" && (challenge.excess ?? 0) === 0 ? "optimal" : challenge.status}</strong>
                  </div>
                  <div>
                    <span>Graph</span>
                    <strong>{challenge.spec.family}</strong>
                  </div>
                </div>
                {challenge.certifiedPath && revealFullBfsText && (
                  <div className="shortest-card">
                    <span>Shortest</span>
                    <strong>{challenge.certifiedPath.generatorIds.join(" ") || "id"}</strong>
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="challenge-board">
            <section className="challenge-lane">
              <div className="challenge-lane-head">
                <div>
                  <h2>User</h2>
                  <p data-testid="user-word" title={challenge ? userWord : "-"}>{challenge ? userWord : "-"}</p>
                </div>
                <div className="lane-side">
                  <div className="lane-badges">
                    <span>{lastUserMove}</span>
                    <span>{legalChallengeEdges.length} moves</span>
                  </div>
                  <div className="legal-move-strip" data-testid="legal-moves">
                    <span className="legal-move-title">Moves</span>
                    {(legalMoveLabels.length ? legalMoveLabels : ["-"]).map((label) => (
                      <span key={label}>{label}</span>
                    ))}
                  </div>
                </div>
              </div>
              <section className="stage challenge-stage">
                <GraphCanvas
                  view={challengeView}
                  testId="challenge-canvas"
                  selectedNodeId={challenge ? currentStateId : null}
                  goalNodeId={challenge ? stateId(challenge.target) : null}
                  pathOverlay={userTrailOverlay}
                  activeEdgeKeys={legalChallengeEdgeKeys}
                  onEdgeClick={onChallengeEdgeClick}
                />
                {!challenge && <div className="game-empty">New</div>}
              </section>
            </section>

            <section className="challenge-lane bfs-lane">
              <div className="challenge-lane-head">
                <div>
                  <h2>BFS</h2>
                  <p data-testid="bfs-word" title={challenge ? bfsWord : "-"}>{challenge ? bfsWord : "-"}</p>
                </div>
                <div className="lane-badges">
                  <span>{shownBfsMove}</span>
                  <span>{bfsStepCount}/{challenge?.optimalLength ?? 0}</span>
                </div>
              </div>
              <section className="stage challenge-stage">
                <GraphCanvas
                  view={challengeView}
                  testId="challenge-bfs-canvas"
                  selectedNodeId={challenge ? bfsCurrentStateId : null}
                  pathOverlay={bfsTrailOverlay}
                />
                {!challenge && <div className="game-empty">New</div>}
              </section>
            </section>
            {error && <div className="error-strip">{error}</div>}
            {busy && <div className="busy-strip">Working</div>}
          </section>
        </section>
      )}
    </main>
  );
}
