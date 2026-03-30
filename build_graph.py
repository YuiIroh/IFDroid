from __future__ import annotations

import argparse
import os
import pickle
from typing import Tuple

import networkx as nx
import pandas as pd


REQUIRED_NODE_COLUMNS = ["node_id", "class_dot", "method_name_norm", "owner_type", "api_text"]
REQUIRED_EDGE_COLUMNS = ["src", "dst"]


def _check_required_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少必要列: {missing}")


def build_graph(
    nodes_csv: str,
    edges_csv: str,
    output_dir: str,
    graph_name: str = "call_graph.pkl",
    undirected: bool = True,
) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    """
    从 nodes_intermediate.csv 和 edges_intermediate.csv 构建函数调用图。

    设计选择：
    1. 默认转为无向简单图，便于复现论文中常见的中心性计算设置。
    2. 节点属性从 nodes_intermediate.csv 中读取并挂到图节点上。
    3. 边按 src-dst 去重；call_count 不参与重复加边。
    """
    os.makedirs(output_dir, exist_ok=True)

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)

    _check_required_columns(nodes_df, REQUIRED_NODE_COLUMNS, "nodes_csv")
    _check_required_columns(edges_df, REQUIRED_EDGE_COLUMNS, "edges_csv")

    nodes_df = nodes_df.copy()
    edges_df = edges_df.copy()

    nodes_df["node_id"] = nodes_df["node_id"].astype(str).str.strip()
    edges_df["src"] = edges_df["src"].astype(str).str.strip()
    edges_df["dst"] = edges_df["dst"].astype(str).str.strip()

    # 构造图
    G = nx.Graph() if undirected else nx.DiGraph()

    # 先加节点并保留原始属性
    attr_cols = list(nodes_df.columns)
    node_records = []
    for row in nodes_df.itertuples(index=False):
        rec = {col: getattr(row, col) for col in attr_cols}
        node_id = str(rec["node_id"]).strip()
        node_records.append(node_id)
        G.add_node(node_id, **rec)

    # 如 edges 中出现了 nodes 表里没有的点，也补入图里，避免丢边
    edge_pairs = []
    missing_nodes = set()
    for row in edges_df[["src", "dst"]].dropna().itertuples(index=False):
        src = str(row.src).strip()
        dst = str(row.dst).strip()
        if src not in G:
            missing_nodes.add(src)
            G.add_node(src, node_id=src, inferred_from_edges=True)
        if dst not in G:
            missing_nodes.add(dst)
            G.add_node(dst, node_id=dst, inferred_from_edges=True)
        edge_pairs.append((src, dst))

    G.add_edges_from(edge_pairs)

    # 保存图
    graph_path = os.path.join(output_dir, graph_name)
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    # 保存简单统计
    summary = pd.DataFrame(
        [
            {
                "graph_type": type(G).__name__,
                "num_nodes": G.number_of_nodes(),
                "num_edges": G.number_of_edges(),
                "nodes_csv_rows": len(nodes_df),
                "edges_csv_rows": len(edges_df),
                "missing_nodes_in_nodes_csv": len(missing_nodes),
                "undirected": undirected,
            }
        ]
    )
    summary.to_csv(os.path.join(output_dir, "graph_summary.csv"), index=False, encoding="utf-8-sig")

    if missing_nodes:
        pd.DataFrame({"missing_node_id": sorted(missing_nodes)}).to_csv(
            os.path.join(output_dir, "missing_nodes_from_edges.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    return G, nodes_df, edges_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建函数调用图并保存为 pickle")
    parser.add_argument("--nodes_csv", required=True)
    parser.add_argument("--edges_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--graph_name", default="call_graph.pkl")
    parser.add_argument("--directed", action="store_true", help="默认无向图；加此参数则构建有向图")
    args = parser.parse_args()

    build_graph(
        nodes_csv=args.nodes_csv,
        edges_csv=args.edges_csv,
        output_dir=args.output_dir,
        graph_name=args.graph_name,
        undirected=not args.directed,
    )
    print("图构建完成")
