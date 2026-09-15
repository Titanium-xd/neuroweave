/**
 * Layout + adjacency for the 150-node MaleCNS v1.0-derived prototype subgraph
 * used in the connectome explorer and the hero field.
 *
 * The node coordinates are an illustrative force-free layout generated
 * deterministically for display purposes — they are not anatomical positions.
 * Replace `buildGraph()` with the real subgraph export when wiring the backend.
 */

export interface Neuron {
  id: number;
  name: string;
  region: string;
  x: number;
  y: number;
  inDegree: number;
  outDegree: number;
  stimulated: boolean;
  active: boolean;
}

export interface Edge {
  id: string;
  source: number;
  target: number;
  weight: number;
}

export interface Graph {
  nodes: Neuron[];
  edges: Edge[];
  stats: {
    nodes: number;
    edges: number;
    density: number;
    meanDegree: number;
    stimulated: number;
    active: number;
  };
}

const REGIONS = [
  "sensory-in",
  "interneuron-a",
  "interneuron-b",
  "recurrent-core",
  "premotor",
] as const;

/** Deterministic PRNG so server and client render identically. */
function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Real graph statistics for the 150-node MaleCNS v1.0 prototype subgraph.
 * Source: artifacts extracted from data/raw/malecns_e2e_validation/
 *
 * nodes : 150   (first 150 neurons in the MaleCNS cache)
 * edges : 3029  (directed synaptic connections within that subgraph)
 * density: 0.1355
 * mean degree: 40.4
 *
 * Node POSITIONS are an illustrative force-free layout generated
 * deterministically for display purposes — they are NOT anatomical positions.
 * No anatomical coordinates are available in the current data export.
 */
export const MALECNS_SUBGRAPH_STATS = {
  nodes: 150,
  edges: 3029,
  density: 0.1355,
  meanDegree: 40.4,
  maxOutDegree: 129,
  maxInDegree: 67,
  source: "MaleCNS v1.0 — adult male Drosophila CNS",
  note: "Illustrative layout only — node positions are not anatomical.",
} as const;

export function buildGraph(nodeCount = 150, seed = 20250915): Graph {
  const rand = mulberry32(seed);
  const nodes: Neuron[] = [];
  const columns = 5;
  const perColumn = Math.ceil(nodeCount / columns);

  for (let i = 0; i < nodeCount; i++) {
    const col = Math.floor(i / perColumn);
    const row = i % perColumn;
    const jitterX = (rand() - 0.5) * 0.11;
    const jitterY = (rand() - 0.5) * 0.05;
    nodes.push({
      id: i,
      name: `n${String(i).padStart(3, "0")}`,
      region: REGIONS[Math.min(col, REGIONS.length - 1)]!,
      x: 0.09 + (col / (columns - 1)) * 0.82 + jitterX,
      y: 0.07 + (row / (perColumn - 1)) * 0.86 + jitterY,
      inDegree: 0,
      outDegree: 0,
      stimulated: false,
      active: false,
    });
  }

  const edges: Edge[] = [];
  const seen = new Set<string>();
  const addEdge = (s: number, t: number) => {
    if (s === t) return;
    const id = `${s}-${t}`;
    if (seen.has(id)) return;
    seen.add(id);
    edges.push({ id, source: s, target: t, weight: 0.25 + rand() * 0.75 });
    nodes[s]!.outDegree++;
    nodes[t]!.inDegree++;
  };

  // Sparse directed layout with recurrent core — mirrors the sparse,
  // directed character of the real MaleCNS connectivity.
  // Edge topology is illustrative; real topology is in the Python backend.
  for (let i = 0; i < nodeCount; i++) {
    const fanout = 1 + Math.floor(rand() * 3);
    for (let k = 0; k < fanout; k++) {
      const forward = Math.floor(rand() * perColumn) + perColumn;
      addEdge(i, (i + forward) % nodeCount);
    }
    if (rand() < 0.22) addEdge(i, Math.floor(rand() * nodeCount));
    if (rand() < 0.14) addEdge(Math.floor(rand() * nodeCount), i);
  }

  const stimulatedIds = nodes.slice(0, perColumn).filter(() => rand() < 0.28);
  stimulatedIds.forEach((n) => {
    n.stimulated = true;
    n.active = true;
  });
  nodes.forEach((n) => {
    if (!n.stimulated && rand() < 0.24) n.active = true;
  });

  // Use real graph statistics for the stats panel
  const stats = {
    nodes: MALECNS_SUBGRAPH_STATS.nodes,
    edges: MALECNS_SUBGRAPH_STATS.edges,
    density: MALECNS_SUBGRAPH_STATS.density,
    meanDegree: MALECNS_SUBGRAPH_STATS.meanDegree,
    stimulated: nodes.filter((n) => n.stimulated).length,
    active: nodes.filter((n) => n.active).length,
  };

  return { nodes, edges, stats };
}

export const REGION_LABELS: Record<string, string> = {
  "sensory-in": "Input population",
  "interneuron-a": "Interneuron layer A",
  "interneuron-b": "Interneuron layer B",
  "recurrent-core": "Recurrent core",
  premotor: "Pre-motor population",
};
