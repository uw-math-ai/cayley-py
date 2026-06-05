import { useEffect, useMemo, useRef, useState, type MouseEvent } from "react";
import type { GraphEdge, GraphNode, GraphView } from "./types";

const PALETTE = [
  [34, 150, 137],
  [238, 137, 42],
  [174, 61, 114],
  [81, 122, 195],
  [112, 160, 72],
  [191, 77, 68],
  [117, 92, 178],
  [35, 130, 178]
];

type Hover =
  | {
      kind: "node";
      node: GraphNode;
      x: number;
      y: number;
    }
  | {
      kind: "edge";
      edge: GraphEdge;
      source?: GraphNode;
      target?: GraphNode;
      x: number;
      y: number;
    };

type HoverEdge = {
  edge: GraphEdge;
  source: GraphNode;
  target: GraphNode;
};

type PathOverlay = {
  targetId: string;
  nodeIds: string[];
  edgeKeys: string[];
  focusEdgeKeys?: string[];
  nodes: GraphNode[];
  edges: GraphEdge[];
};

function colorFor(generatorId: string): number[] {
  let hash = 0;
  for (const char of generatorId) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return PALETTE[hash % PALETTE.length];
}

function compile(gl: WebGLRenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("shader");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader) || "shader compile");
  }
  return shader;
}

const VERTEX_SOURCE = `
attribute vec2 a_position;
attribute vec3 a_color;
uniform float u_point_size;
varying vec3 v_color;
void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  gl_PointSize = u_point_size;
  v_color = a_color;
}`;

function program(gl: WebGLRenderingContext, fragmentSource: string): WebGLProgram {
  const vertex = compile(gl, gl.VERTEX_SHADER, VERTEX_SOURCE);
  const fragment = compile(gl, gl.FRAGMENT_SHADER, fragmentSource);
  const out = gl.createProgram();
  if (!out) throw new Error("program");
  gl.attachShader(out, vertex);
  gl.attachShader(out, fragment);
  gl.linkProgram(out);
  if (!gl.getProgramParameter(out, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(out) || "program link");
  }
  return out;
}

function edgeKey(edge: GraphEdge): string {
  return `${edge.source}|${edge.target}|${edge.generatorId}`;
}

export default function GraphCanvas({
  view,
  testId,
  selectedNodeId,
  goalNodeId,
  pathOverlay,
  activeEdgeKeys,
  onNodeClick,
  onEdgeClick
}: {
  view: GraphView | null;
  testId?: string;
  selectedNodeId?: string | null;
  goalNodeId?: string | null;
  pathOverlay?: PathOverlay | null;
  activeEdgeKeys?: string[];
  onNodeClick?: (node: GraphNode | null) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const [viewport, setViewport] = useState({ width: 0, height: 0 });

  const nodeMap = useMemo(() => {
    const map = new Map<string, GraphNode>();
    view?.nodes.forEach((node) => map.set(node.id, node));
    return map;
  }, [view]);

  const renderNodeMap = useMemo(() => {
    const map = new Map(nodeMap);
    pathOverlay?.nodes.forEach((node) => map.set(node.id, node));
    return map;
  }, [nodeMap, pathOverlay]);

  const hoverEdges = useMemo(() => {
    const edges = [...(pathOverlay?.edges ?? []), ...(view?.edges ?? [])];
    const seen = new Set<string>();
    const out: HoverEdge[] = [];
    for (const edge of edges) {
      const key = edgeKey(edge);
      if (seen.has(key)) continue;
      seen.add(key);
      const source = renderNodeMap.get(edge.source);
      const target = renderNodeMap.get(edge.target);
      if (source && target) out.push({ edge, source, target });
    }
    return out;
  }, [view, pathOverlay, renderNodeMap]);

  const activeEdgeSet = useMemo(() => new Set(activeEdgeKeys ?? []), [activeEdgeKeys]);
  const activeEdges = useMemo(
    () => hoverEdges.filter((candidate) => activeEdgeSet.has(edgeKey(candidate.edge))),
    [hoverEdges, activeEdgeSet]
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const updateSize = () => {
      const rect = canvas.getBoundingClientRect();
      setViewport((current) =>
        Math.abs(current.width - rect.width) < 0.5 && Math.abs(current.height - rect.height) < 0.5
          ? current
          : { width: rect.width, height: rect.height }
      );
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !view) return;
    const gl = canvas.getContext("webgl", { antialias: true, preserveDrawingBuffer: true });
    if (!gl) return;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.055, 0.054, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const lineProg = program(
      gl,
      `
      precision mediump float;
      varying vec3 v_color;
      void main() {
        gl_FragColor = vec4(v_color, 0.52);
      }`
    );
    const pointProg = program(
      gl,
      `
      precision mediump float;
      varying vec3 v_color;
      void main() {
        vec2 d = gl_PointCoord - vec2(0.5, 0.5);
        if (length(d) > 0.5) discard;
        gl_FragColor = vec4(v_color, 0.96);
      }`
    );

    const edgeFloats: number[] = [];
    const dimBase = Boolean(pathOverlay) || activeEdgeSet.size > 0;
    const pathNodeIds = new Set(pathOverlay?.nodeIds ?? []);
    const pushEdge = (data: number[], source: GraphNode, target: GraphNode, color: number[]) => {
      const loop = source.id === target.id;
      const sourceX = loop ? source.x - 0.035 : source.x;
      const sourceY = loop ? source.y + 0.045 : source.y;
      const targetX = loop ? target.x + 0.035 : target.x;
      const targetY = loop ? target.y + 0.045 : target.y;
      data.push(sourceX, sourceY, color[0], color[1], color[2]);
      data.push(targetX, targetY, color[0], color[1], color[2]);
    };
    for (const edge of view.edges) {
      const source = renderNodeMap.get(edge.source);
      const target = renderNodeMap.get(edge.target);
      if (!source || !target) continue;
      const active = activeEdgeSet.has(edgeKey(edge));
      const color = active ? [0.23, 0.9, 0.78] : dimBase ? [0.16, 0.16, 0.15] : colorFor(edge.generatorId).map((c) => c / 255);
      pushEdge(edgeFloats, source, target, color);
    }

    const overlayEdgeFloats: number[] = [];
    const overlayFocusEdgeFloats: number[] = [];
    const overlayEdges = pathOverlay?.edges ?? [];
    const overlayKeys = new Set(pathOverlay?.edgeKeys ?? overlayEdges.map(edgeKey));
    const overlayFocusKeys = new Set(pathOverlay?.focusEdgeKeys ?? []);
    for (const edge of [...view.edges, ...overlayEdges]) {
      const key = edgeKey(edge);
      if (!overlayKeys.has(key)) continue;
      const source = renderNodeMap.get(edge.source);
      const target = renderNodeMap.get(edge.target);
      if (!source || !target) continue;
      pushEdge(
        overlayFocusKeys.has(key) ? overlayFocusEdgeFloats : overlayEdgeFloats,
        source,
        target,
        overlayFocusKeys.has(key) ? [0.23, 0.9, 0.78] : [1.0, 0.74, 0.18]
      );
    }

    const draw = (prog: WebGLProgram, data: number[], mode: number, pointSize: number) => {
      if (!data.length) return;
      gl.useProgram(prog);
      const positionLoc = gl.getAttribLocation(prog, "a_position");
      const colorLoc = gl.getAttribLocation(prog, "a_color");
      const pointSizeLoc = gl.getUniformLocation(prog, "u_point_size");
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(positionLoc);
      gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 20, 0);
      gl.enableVertexAttribArray(colorLoc);
      gl.vertexAttribPointer(colorLoc, 3, gl.FLOAT, false, 20, 8);
      gl.uniform1f(pointSizeLoc, pointSize * dpr);
      gl.drawArrays(mode, 0, data.length / 5);
    };

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.lineWidth(1);
    draw(lineProg, edgeFloats, gl.LINES, 1);
    draw(lineProg, overlayEdgeFloats, gl.LINES, 1);
    draw(lineProg, overlayFocusEdgeFloats, gl.LINES, 1);

    const nodeFloats: number[] = [];
    for (const node of view.nodes) {
      const depth = node.distance ?? 0;
      const tone = Math.min(1, 0.25 + depth / Math.max(1, view.metadata.diameter) * 0.75);
      if (pathNodeIds.has(node.id)) {
        nodeFloats.push(node.x, node.y, 1.0, 0.77, 0.24);
      } else if (dimBase) {
        nodeFloats.push(node.x, node.y, 0.22, 0.21, 0.19);
      } else {
        nodeFloats.push(node.x, node.y, 0.9, 0.88 - tone * 0.25, 0.62 + tone * 0.28);
      }
    }
    draw(pointProg, nodeFloats, gl.POINTS, view.nodes.length > 3000 ? 3 : 7);

    const overlayNodeFloats: number[] = [];
    for (const node of pathOverlay?.nodes ?? []) {
      if (nodeMap.has(node.id)) continue;
      overlayNodeFloats.push(node.x, node.y, 1.0, 0.77, 0.24);
    }
    draw(pointProg, overlayNodeFloats, gl.POINTS, view.nodes.length > 3000 ? 5 : 9);

    if (goalNodeId) {
      const goal = renderNodeMap.get(goalNodeId);
      if (goal) {
        draw(pointProg, [goal.x, goal.y, 1.0, 0.28, 0.18], gl.POINTS, view.nodes.length > 3000 ? 10 : 18);
      }
    }

    if (selectedNodeId) {
      const selected = renderNodeMap.get(selectedNodeId);
      if (selected) {
        draw(pointProg, [selected.x, selected.y, 0.98, 0.96, 0.84], gl.POINTS, view.nodes.length > 3000 ? 7 : 13);
      }
    }
  }, [view, nodeMap, renderNodeMap, pathOverlay, selectedNodeId, goalNodeId, activeEdgeSet, viewport]);

  const pointFor = (event: MouseEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      rect,
      localX: event.clientX - rect.left,
      localY: event.clientY - rect.top
    };
  };

  const nodePoint = (node: GraphNode, rect: DOMRect) => ({
    x: ((node.x + 1) / 2) * rect.width,
    y: ((1 - node.y) / 2) * rect.height
  });

  const segmentDistance = (
    px: number,
    py: number,
    ax: number,
    ay: number,
    bx: number,
    by: number
  ) => {
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq === 0) return Math.hypot(px - ax, py - ay);
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSq));
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  };

  const pickNode = (event: MouseEvent<HTMLCanvasElement>, thresholdPx = 12): GraphNode | null => {
    if (!view) return null;
    const { rect, localX, localY } = pointFor(event);
    let best: GraphNode | null = null;
    let bestDistance = thresholdPx;
    for (const node of view.nodes) {
      const point = nodePoint(node, rect);
      const distance = Math.hypot(point.x - localX, point.y - localY);
      if (distance < bestDistance) {
        best = node;
        bestDistance = distance;
      }
    }
    return best;
  };

  const pickEdge = (event: MouseEvent<HTMLCanvasElement>, thresholdPx = 7): HoverEdge | null => {
    if (!view) return null;
    const { rect, localX, localY } = pointFor(event);
    let best: HoverEdge | null = null;
    let bestDistance = thresholdPx;
    for (const candidate of hoverEdges) {
      const source = nodePoint(candidate.source, rect);
      const target = nodePoint(candidate.target, rect);
      const loop = candidate.source.id === candidate.target.id;
      const distance = loop
        ? Math.hypot(localX - source.x, localY - source.y)
        : segmentDistance(localX, localY, source.x, source.y, target.x, target.y);
      if (onEdgeClick && activeEdgeSet.size > 0 && !activeEdgeSet.has(edgeKey(candidate.edge))) continue;
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    return best;
  };

  const onMove = (event: MouseEvent<HTMLCanvasElement>) => {
    const { rect } = pointFor(event);
    if (onEdgeClick) {
      const edge = pickEdge(event, 10);
      if (edge) {
        setHover({
          kind: "edge",
          edge: edge.edge,
          source: edge.source,
          target: edge.target,
          x: event.clientX - rect.left,
          y: event.clientY - rect.top
        });
        return;
      }
    }
    const node = pickNode(event);
    if (node) {
      setHover({ kind: "node", node, x: event.clientX - rect.left, y: event.clientY - rect.top });
      return;
    }
    const edge = pickEdge(event);
    setHover(
      edge
        ? {
            kind: "edge",
            edge: edge.edge,
            source: edge.source,
            target: edge.target,
            x: event.clientX - rect.left,
            y: event.clientY - rect.top
          }
        : null
    );
  };

  const onClick = (event: MouseEvent<HTMLCanvasElement>) => {
    if (onEdgeClick) {
      const edge = pickEdge(event, 12);
      if (edge) {
        onEdgeClick(edge.edge);
        return;
      }
    }
    const selected = pickNode(event, 14);
    if (onNodeClick) onNodeClick(selected);
  };

  const screenPoint = (node: GraphNode) => ({
    x: ((node.x + 1) / 2) * viewport.width,
    y: ((1 - node.y) / 2) * viewport.height
  });

  const goalNode = goalNodeId ? renderNodeMap.get(goalNodeId) : undefined;
  const goalMarker = goalNode && viewport.width > 0 && viewport.height > 0 ? (() => {
    const point = screenPoint(goalNode);
    const labelWidth = 56;
    return {
      x: point.x,
      y: point.y,
      labelX: Math.max(8, Math.min(viewport.width - labelWidth - 8, point.x - labelWidth / 2)),
      labelY: Math.max(12, Math.min(viewport.height - 32, point.y - 42)),
      labelWidth
    };
  })() : null;

  const legalEdgeOverlays = activeEdges.map((candidate) => {
    const source = screenPoint(candidate.source);
    const target = screenPoint(candidate.target);
    const label = candidate.edge.generatorLabel || candidate.edge.generatorId;
    const labelWidth = Math.max(30, Math.min(94, label.length * 7 + 16));
    const isLoop = candidate.source.id === candidate.target.id;
    if (isLoop) {
      const labelX = Math.max(8, Math.min(viewport.width - labelWidth - 8, source.x - labelWidth / 2));
      const labelY = Math.max(12, source.y - 68);
      return {
        key: edgeKey(candidate.edge),
        label,
        labelWidth,
        labelX,
        labelY,
        source,
        target,
        isLoop: true as const,
        path: `M ${source.x - 18} ${source.y - 12} C ${source.x - 48} ${source.y - 62}, ${source.x + 48} ${source.y - 62}, ${source.x + 18} ${source.y - 12}`
      };
    }
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const length = Math.hypot(dx, dy) || 1;
    const offset = Math.min(20, length * 0.25);
    const x1 = source.x + (dx / length) * offset;
    const y1 = source.y + (dy / length) * offset;
    const x2 = target.x - (dx / length) * offset;
    const y2 = target.y - (dy / length) * offset;
    const labelX = Math.max(8, Math.min(viewport.width - labelWidth - 8, (x1 + x2) / 2 - labelWidth / 2));
    const labelY = Math.max(12, Math.min(viewport.height - 28, (y1 + y2) / 2 - 14));
    return {
      key: edgeKey(candidate.edge),
      label,
      labelWidth,
      labelX,
      labelY,
      source,
      target,
      isLoop: false as const,
      x1,
      y1,
      x2,
      y2
    };
  });

  return (
    <div className={`canvas-wrap${onNodeClick || onEdgeClick ? " is-clickable" : ""}`}>
      <canvas data-testid={testId} ref={canvasRef} onClick={onClick} onMouseMove={onMove} onMouseLeave={() => setHover(null)} />
      {activeEdges.length > 0 && viewport.width > 0 && viewport.height > 0 && (
        <svg className="legal-move-overlay" viewBox={`0 0 ${viewport.width} ${viewport.height}`} aria-hidden="true">
          <defs>
            <marker id={`arrow-${testId ?? "graph"}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" />
            </marker>
          </defs>
          {legalEdgeOverlays.map((item) => (
            <g key={`legal-${item.key}`} className="legal-move-edge">
              <circle className="legal-source-halo" cx={item.source.x} cy={item.source.y} r="16" />
              <circle className="legal-target-halo" cx={item.target.x} cy={item.target.y} r="13" />
              {item.isLoop ? (
                <path className="legal-edge-path" d={item.path} markerEnd={`url(#arrow-${testId ?? "graph"})`} />
              ) : (
                <line className="legal-edge-path" x1={item.x1} y1={item.y1} x2={item.x2} y2={item.y2} markerEnd={`url(#arrow-${testId ?? "graph"})`} />
              )}
              <rect className="legal-edge-label-bg" x={item.labelX} y={item.labelY} width={item.labelWidth} height="24" rx="7" />
              <text className="legal-edge-label" x={item.labelX + item.labelWidth / 2} y={item.labelY + 16}>
                {item.label}
              </text>
            </g>
          ))}
        </svg>
      )}
      {goalMarker && (
        <svg className="goal-marker-overlay" viewBox={`0 0 ${viewport.width} ${viewport.height}`} aria-hidden="true">
          <g className="goal-marker" data-testid={`${testId ?? "graph"}-goal-marker`}>
            <circle className="goal-marker-pulse" cx={goalMarker.x} cy={goalMarker.y} r="23" />
            <circle className="goal-marker-ring" cx={goalMarker.x} cy={goalMarker.y} r="15" />
            <line className="goal-marker-cross" x1={goalMarker.x - 22} y1={goalMarker.y} x2={goalMarker.x + 22} y2={goalMarker.y} />
            <line className="goal-marker-cross" x1={goalMarker.x} y1={goalMarker.y - 22} x2={goalMarker.x} y2={goalMarker.y + 22} />
            <rect className="goal-marker-label-bg" x={goalMarker.labelX} y={goalMarker.labelY} width={goalMarker.labelWidth} height="24" rx="7" />
            <text className="goal-marker-label" x={goalMarker.labelX + goalMarker.labelWidth / 2} y={goalMarker.labelY + 16}>
              Goal
            </text>
          </g>
        </svg>
      )}
      {!view && <div className="empty-canvas">Load</div>}
      {view && (
        <div className="canvas-hud">
          <span>{view.kind}</span>
          <span>{view.nodes.length.toLocaleString()} nodes</span>
          <span>{view.edges.length.toLocaleString()} edges</span>
          {view.certified && <span>certified</span>}
          {view.truncated && <span>sampled</span>}
        </div>
      )}
      {hover && (
        <div className="tooltip" style={{ left: hover.x + 12, top: hover.y + 12 }}>
          {hover.kind === "node" ? (
            <>
              <strong>{hover.node.label}</strong>
              <span>d={hover.node.distance ?? "?"}</span>
            </>
          ) : (
            <>
              <strong>{hover.edge.generatorLabel}</strong>
              <span>
                {hover.source?.label ?? hover.edge.source}
                {" -> "}
                {hover.target?.label ?? hover.edge.target}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}
