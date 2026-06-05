export type GraphSpec = {
  family: string;
  n: number;
  params: Record<string, number | string>;
  inversePolicy: "default" | "listed" | "closed";
};

export type FamilyParam = {
  id: string;
  label: string;
  type: "integer" | "select";
  default: number;
  min?: number;
  options?: number[];
};

export type FamilyDef = {
  id: string;
  label: string;
  minN: number;
  defaultInversePolicy: string;
  description: string;
  parameters: FamilyParam[];
};

export type GeneratorSpec = {
  id: string;
  label: string;
  permutation: number[];
  involutive: boolean;
  inverseOf?: string | null;
};

export type Summary = {
  specHash: string;
  n: number;
  stateSpace: string;
  startState: number[];
  generatorCount: number;
  estimatedGroupUpperBound: number;
  generators: GeneratorSpec[];
  exactCached: boolean;
  exact?: {
    nStates: number;
    diameter: number;
    layerSizes: number[];
  } | null;
};

export type GraphNode = {
  id: string;
  state: number[];
  label: string;
  distance: number | null;
  x: number;
  y: number;
};

export type GraphEdge = {
  source: string;
  target: string;
  generatorId: string;
  generatorLabel: string;
  forwardLayer: boolean;
};

export type GraphView = {
  kind: string;
  layout: string;
  certified: boolean;
  specHash: string;
  metadata: {
    nStates: number;
    diameter: number;
    layerSizes: number[];
  };
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
  limitExceeded?: {
    cap: number;
    visited: number;
  };
};

export type ShortestPaths = {
  certified: boolean;
  specHash: string;
  layout: string;
  start: number[];
  target: number[];
  targetId: string;
  length: number;
  pathCount: number;
  pathCountCapped: boolean;
  canonicalPath: CertifiedPath;
  nodeIds: string[];
  edgeKeys: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  truncated: boolean;
};

export type CertifiedPath = {
  target: number[];
  generatorIds: string[];
  states: number[][];
  length: number;
  certified: boolean;
};

export type ChallengeState = {
  sessionId: string;
  spec: GraphSpec;
  specHash: string;
  target: number[];
  current: number[];
  start: number[];
  userStates: number[][];
  targetDistance: number;
  userPath: string[];
  userLength: number;
  status: "active" | "completed" | "forfeited";
  certifiedPath: CertifiedPath | null;
  optimalLength: number | null;
  excess: number | null;
  generators: GeneratorSpec[];
};
