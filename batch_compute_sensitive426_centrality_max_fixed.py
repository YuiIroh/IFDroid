#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fixed version:
- Compatible with scipy csr_array / csr_matrix differences
- Still scans all APK folders
- Builds FULL graph
- Scores only sensitive-API target nodes
- Aggregates overloaded functions by MAX
- Outputs 426x4 per APK
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


# =========================================================
# Hard-coded paths
# =========================================================
INPUT_ROOT = r"E:\Text to image\IFDroid\APK_graphs_output"
OUTPUT_ROOT = r"E:\Text to image\IFDroid\APK_sensitive426_output"
API_SET_CSV = r"E:\Text to image\IFDroid\API Set.csv"

NODES_FILENAME = "nodes_api_named.csv"
EDGES_FILENAME = "edges_api_named.csv"

USE_UNDIRECTED_GRAPH = True
KATZ_ALPHA = 0.5
KATZ_BETA = 1.0

WRITE_GML = False
WRITE_JSON = False


def safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_api_union(api_set_csv: str) -> List[str]:
    df = pd.read_csv(api_set_csv)
    if "Union" not in df.columns:
        raise ValueError(f"'Union' column not found in API Set file: {api_set_csv}")

    union = (
        df["Union"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    union = union[union != ""].tolist()

    seen = set()
    ordered = []
    for x in union:
        if x not in seen:
            seen.add(x)
            ordered.append(x)
    return ordered


def build_full_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, undirected: bool = True):
    G = nx.Graph() if undirected else nx.DiGraph()

    for row in nodes_df.itertuples(index=False):
        node_id = getattr(row, "node_id")
        if pd.isna(node_id):
            continue

        G.add_node(
            node_id,
            api_text=getattr(row, "api_text", ""),
            method_key=getattr(row, "method_key", ""),
            class_dot=getattr(row, "class_dot", ""),
            method_name_norm=getattr(row, "method_name_norm", ""),
            descriptor_compact=getattr(row, "descriptor_compact", ""),
            owner_type=getattr(row, "owner_type", ""),
            external_bool=bool(getattr(row, "external_bool", False)),
            entrypoint_bool=bool(getattr(row, "entrypoint_bool", False)),
            api_union_exact_match=bool(getattr(row, "api_union_exact_match", False)),
            matched_api_name=getattr(row, "matched_api_name", ""),
        )

    has_count = "call_count" in edges_df.columns

    for row in edges_df.itertuples(index=False):
        src = getattr(row, "src")
        dst = getattr(row, "dst")

        if pd.isna(src) or pd.isna(dst):
            continue

        call_count = getattr(row, "call_count", 1) if has_count else 1
        if pd.isna(call_count):
            call_count = 1
        call_count = int(call_count)

        if G.has_edge(src, dst):
            G[src][dst]["call_count"] = G[src][dst].get("call_count", 1) + call_count
        else:
            G.add_edge(src, dst, call_count=call_count)

    return G


def choose_api_name_column(nodes_df: pd.DataFrame) -> pd.Series:
    api_text = nodes_df["api_text"].apply(safe_str) if "api_text" in nodes_df.columns else pd.Series([""] * len(nodes_df))
    matched = nodes_df["matched_api_name"].apply(safe_str) if "matched_api_name" in nodes_df.columns else pd.Series([""] * len(nodes_df))
    exact = nodes_df["api_union_exact_match"] if "api_union_exact_match" in nodes_df.columns else pd.Series([False] * len(nodes_df))

    chosen = api_text.copy()
    mask = exact.fillna(False) & (matched != "")
    chosen.loc[mask] = matched.loc[mask]
    return chosen


def get_target_nodes(nodes_df: pd.DataFrame, api_union_order: List[str]) -> Tuple[pd.DataFrame, Set[str]]:
    api_union_set = set(api_union_order)

    df = nodes_df.copy()
    df["api_name_for_set"] = choose_api_name_column(df)

    target_df = df[df["api_name_for_set"].isin(api_union_set)].copy()
    target_df = target_df[target_df["node_id"].notna()].copy()
    target_df["node_id"] = target_df["node_id"].astype(str)

    return target_df, set(target_df["node_id"].tolist())


def save_graph_json(G: nx.Graph, out_json: Path) -> None:
    data = nx.node_link_data(G)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def degree_centrality_for_targets(G: nx.Graph, target_nodes: List[str]) -> Dict[str, float]:
    n = G.number_of_nodes()
    denom = max(n - 1, 1)
    return {node: float(G.degree(node) / denom) for node in target_nodes}


def katz_twohop_for_targets(G: nx.Graph, target_nodes: List[str], alpha: float = 0.5, beta: float = 1.0) -> Dict[str, float]:
    """
    x = beta * (A1) + beta * alpha * (A^2 1)

    Fix:
    Convert scipy sparse array -> csr_matrix, then use getrow safely.
    """
    nodelist = list(G.nodes())
    if not nodelist:
        return {}

    index_of = {node: i for i, node in enumerate(nodelist)}

    A = nx.to_scipy_sparse_array(
        G,
        nodelist=nodelist,
        dtype=np.float64,
        weight=None,
        format="csr",
    )

    # Critical compatibility fix:
    A = sparse.csr_matrix(A)

    ones = np.ones((A.shape[0],), dtype=np.float64)
    one_hop = A @ ones

    out = {}
    for node in target_nodes:
        i = index_of[node]
        row = A.getrow(i)
        two_hop_i = float((row @ one_hop).item())
        score = beta * float(one_hop[i]) + beta * alpha * two_hop_i
        out[node] = score
    return out


def closeness_centrality_for_targets(G: nx.Graph, target_nodes: List[str], wf_improved: bool = True) -> Dict[str, float]:
    N = G.number_of_nodes()
    out = {}

    for u in target_nodes:
        sp = nx.single_source_shortest_path_length(G, u)
        totsp = sum(sp.values())
        reachable = len(sp)

        if totsp > 0.0 and N > 1 and reachable > 1:
            c = (reachable - 1.0) / totsp
            if wf_improved:
                c *= (reachable - 1.0) / (N - 1.0)
        else:
            c = 0.0
        out[u] = float(c)

    return out


def harmonic_centrality_for_targets(G: nx.Graph, target_nodes: List[str]) -> Dict[str, float]:
    out = {}
    for u in target_nodes:
        sp = nx.single_source_shortest_path_length(G, u)
        score = 0.0
        for v, d in sp.items():
            if v == u or d <= 0:
                continue
            score += 1.0 / d
        out[u] = float(score)
    return out


def aggregate_targets_to_426(target_detail_df: pd.DataFrame, api_union_order: List[str]) -> pd.DataFrame:
    if target_detail_df.empty:
        rows = []
        for api in api_union_order:
            rows.append({
                "api_name": api,
                "matched_node_count": 0,
                "degree_centrality": 0.0,
                "katz_twohop_centrality": 0.0,
                "closeness_centrality": 0.0,
                "harmonic_centrality": 0.0,
            })
        return pd.DataFrame(rows)

    agg = (
        target_detail_df
        .groupby("api_name_for_set", as_index=False)
        .agg({
            "node_id": "count",
            "degree_centrality": "max",
            "katz_twohop_centrality": "max",
            "closeness_centrality": "max",
            "harmonic_centrality": "max",
        })
        .rename(columns={"node_id": "matched_node_count"})
    )

    agg_map = {row["api_name_for_set"]: row for _, row in agg.iterrows()}

    rows = []
    for api in api_union_order:
        if api in agg_map:
            row = agg_map[api]
            rows.append({
                "api_name": api,
                "matched_node_count": int(row["matched_node_count"]),
                "degree_centrality": float(row["degree_centrality"]),
                "katz_twohop_centrality": float(row["katz_twohop_centrality"]),
                "closeness_centrality": float(row["closeness_centrality"]),
                "harmonic_centrality": float(row["harmonic_centrality"]),
            })
        else:
            rows.append({
                "api_name": api,
                "matched_node_count": 0,
                "degree_centrality": 0.0,
                "katz_twohop_centrality": 0.0,
                "closeness_centrality": 0.0,
                "harmonic_centrality": 0.0,
            })

    return pd.DataFrame(rows)


def compute_one_apk(sample_dir: Path, output_root: Path, api_union_order: List[str]) -> Optional[Dict[str, object]]:
    nodes_csv = sample_dir / NODES_FILENAME
    edges_csv = sample_dir / EDGES_FILENAME

    if not nodes_csv.exists() or not edges_csv.exists():
        return None

    out_dir = output_root / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    required_node_cols = {"node_id", "api_text", "owner_type"}
    required_edge_cols = {"src", "dst"}

    miss_node = required_node_cols - set(nodes_df.columns)
    miss_edge = required_edge_cols - set(edges_df.columns)

    if miss_node:
        raise ValueError(f"nodes file missing required columns: {sorted(miss_node)}")
    if miss_edge:
        raise ValueError(f"edges file missing required columns: {sorted(miss_edge)}")

    G = build_full_graph(nodes_df, edges_df, undirected=USE_UNDIRECTED_GRAPH)

    if WRITE_GML:
        nx.write_gml(G, out_dir / "graph_full.gml")
    if WRITE_JSON:
        save_graph_json(G, out_dir / "graph_full.json")

    target_nodes_df, target_node_set = get_target_nodes(nodes_df, api_union_order)
    target_nodes = sorted(target_node_set)

    if not target_nodes:
        zero_426 = aggregate_targets_to_426(pd.DataFrame(), api_union_order)
        zero_426.to_csv(out_dir / "api426_centrality_max.csv", index=False, encoding="utf-8-sig")
        zero_426[[
            "degree_centrality",
            "katz_twohop_centrality",
            "closeness_centrality",
            "harmonic_centrality",
        ]].to_csv(out_dir / "api426_matrix_only.csv", index=False, encoding="utf-8-sig")

        summary = {
            "apk_folder": sample_dir.name,
            "full_graph_nodes": int(G.number_of_nodes()),
            "full_graph_edges": int(G.number_of_edges()),
            "target_node_count": 0,
            "matched_api_count": 0,
            "api_set_size": int(len(api_union_order)),
            "aggregation_mode": "max",
            "katz_mode": "truncated_2hop",
            "output_426_csv": str(out_dir / "api426_centrality_max.csv"),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    degree_map = degree_centrality_for_targets(G, target_nodes)
    katz_map = katz_twohop_for_targets(G, target_nodes, alpha=KATZ_ALPHA, beta=KATZ_BETA)
    closeness_map = closeness_centrality_for_targets(G, target_nodes, wf_improved=True)
    harmonic_map = harmonic_centrality_for_targets(G, target_nodes)

    target_cent_df = pd.DataFrame({
        "node_id": target_nodes,
        "degree_centrality": [degree_map[n] for n in target_nodes],
        "katz_twohop_centrality": [katz_map[n] for n in target_nodes],
        "closeness_centrality": [closeness_map[n] for n in target_nodes],
        "harmonic_centrality": [harmonic_map[n] for n in target_nodes],
    })

    keep_cols = [
        "node_id",
        "api_name_for_set",
        "api_text",
        "matched_api_name",
        "method_key",
        "class_dot",
        "method_name_norm",
        "descriptor_compact",
        "owner_type",
        "external_bool",
        "entrypoint_bool",
        "api_union_exact_match",
    ]
    keep_cols = [c for c in keep_cols if c in target_nodes_df.columns]

    target_detail_df = target_nodes_df[keep_cols].merge(target_cent_df, on="node_id", how="left")
    target_detail_df.to_csv(out_dir / "target_sensitive_nodes_detail.csv", index=False, encoding="utf-8-sig")

    api426_df = aggregate_targets_to_426(target_detail_df, api_union_order)
    api426_df.to_csv(out_dir / "api426_centrality_max.csv", index=False, encoding="utf-8-sig")

    api426_df[[
        "degree_centrality",
        "katz_twohop_centrality",
        "closeness_centrality",
        "harmonic_centrality",
    ]].to_csv(out_dir / "api426_matrix_only.csv", index=False, encoding="utf-8-sig")

    mapping_rows = []
    for api, sub in target_detail_df.groupby("api_name_for_set"):
        node_ids = sorted(sub["node_id"].astype(str).unique().tolist())
        method_keys = sorted(sub["method_key"].astype(str).fillna("").unique().tolist()) if "method_key" in sub.columns else []
        mapping_rows.append({
            "api_name": api,
            "matched_node_count": len(node_ids),
            "matched_node_ids": " | ".join(node_ids),
            "matched_method_keys": " | ".join(method_keys),
        })
    mapping_df = pd.DataFrame(mapping_rows).sort_values(by=["api_name"])
    mapping_df.to_csv(out_dir / "api_to_node_mapping.csv", index=False, encoding="utf-8-sig")

    summary = {
        "apk_folder": sample_dir.name,
        "input_nodes_csv": str(nodes_csv),
        "input_edges_csv": str(edges_csv),
        "graph_type": "undirected" if USE_UNDIRECTED_GRAPH else "directed",
        "full_graph_nodes": int(G.number_of_nodes()),
        "full_graph_edges": int(G.number_of_edges()),
        "target_node_count": int(len(target_nodes)),
        "matched_api_count": int((api426_df["matched_node_count"] > 0).sum()),
        "api_set_size": int(len(api_union_order)),
        "aggregation_mode": "max",
        "katz_mode": "truncated_2hop",
        "katz_formula": "beta*(A1) + beta*alpha*(A^2 1)",
        "katz_alpha": KATZ_ALPHA,
        "katz_beta": KATZ_BETA,
        "output_target_detail_csv": str(out_dir / "target_sensitive_nodes_detail.csv"),
        "output_426_csv": str(out_dir / "api426_centrality_max.csv"),
        "output_426_matrix_csv": str(out_dir / "api426_matrix_only.csv"),
        "output_mapping_csv": str(out_dir / "api_to_node_mapping.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {input_root}")

    api_union_order = load_api_union(API_SET_CSV)

    sample_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: p.name.lower())

    rows = []
    for sample_dir in sample_dirs:
        try:
            row = compute_one_apk(sample_dir, output_root, api_union_order)
            if row is None:
                print(f"[SKIP] {sample_dir.name}: missing {NODES_FILENAME} or {EDGES_FILENAME}")
            else:
                rows.append(row)
                print(f"[OK] {sample_dir.name}")
        except Exception as e:
            print(f"[ERROR] {sample_dir.name}: {e}")

    if rows:
        pd.DataFrame(rows).to_csv(output_root / "batch_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\nDone. Batch summary saved to: {output_root / 'batch_summary.csv'}")
    else:
        print("\nNo valid APK folders were processed.")


if __name__ == "__main__":
    main()
