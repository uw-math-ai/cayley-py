import type { ChallengeState, FamilyDef, GraphSpec, GraphView, ShortestPaths, Summary } from "./types";

async function post<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text);
  }
  return response.json() as Promise<T>;
}

export async function fetchFamilies(): Promise<FamilyDef[]> {
  const response = await fetch("/api/families");
  if (!response.ok) throw new Error(await response.text());
  const data = (await response.json()) as { families: FamilyDef[] };
  return data.families;
}

export function fetchSummary(spec: GraphSpec): Promise<Summary> {
  return post<Summary>("/api/graph/summary", { spec });
}

export function fetchView(
  spec: GraphSpec,
  mode: string,
  options: { layout?: string; focusState?: number[]; targetState?: number[]; pinnedStates?: number[][]; radius?: number } = {}
): Promise<GraphView> {
  return post<GraphView>("/api/graph/view", {
    spec,
    mode,
    layout: options.layout ?? "layers",
    focusState: options.focusState,
    targetState: options.targetState,
    pinnedStates: options.pinnedStates,
    radius: options.radius ?? 2
  });
}

export function fetchShortestPaths(spec: GraphSpec, targetState: number[], layout: string): Promise<ShortestPaths> {
  return post<ShortestPaths>("/api/graph/shortest-paths", { spec, targetState, layout });
}

export function startChallenge(spec: GraphSpec, difficulty: string): Promise<ChallengeState> {
  return post<ChallengeState>("/api/challenge/start", { spec, difficulty });
}

export function moveChallenge(sessionId: string, generatorId: string): Promise<ChallengeState> {
  return post<ChallengeState>("/api/challenge/move", { sessionId, generatorId });
}

export function forfeitChallenge(sessionId: string): Promise<ChallengeState> {
  return post<ChallengeState>("/api/challenge/forfeit", { sessionId });
}
