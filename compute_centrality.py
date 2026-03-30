from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Iterable, Optional

import networkx as nx
import numpy as np
import pandas as pd

try:
    from scipy import sparse
    from scipy.sparse.linalg import eigsh, spsolve
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


CENTRALITY_COLUMNS = ["degree_centrality", "katz_centrality", "closeness_centrality", "harmonic_centrality"]



def _safe_api_key(class_dot: object, method_name_norm: object) -> str:
    return f"{str(class_dot).strip()} {str(method_name_norm).strip()}".strip()



def _estimate_largest_eigenvalue(G: nx.Graph) -> float:
    """
    估计邻接矩阵最大特征值，用来自动设置更稳健的 Katz alpha。
    对当前这类 1~2 万节点、几万边的图通常够用。
    """
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return 1.0

    if SCIPY_AVAILABLE:
        A = nx.to_scipy_sparse_array(G, dtype=float, format="csr")
        try:
            val = eigsh(A, k=1, which="LM", return_eigenvectors=False)[0]
            val = float(np.abs(val))
            return max(val, 1.0)
        except Exception:
            pass

    # scipy 不可用或失败时的保守兜底：用最大度近似上界
    max_deg = max(dict(G.degree()).values(), default=1)
    return float(max(max_deg, 1))



def _compute_katz(G: nx.Graph, alpha: Optional[float] = None) -> Dict[str, float]:
    lambda_max = _estimate_largest_eigenvalue(G)
    if alpha is None:
        # alpha 必须严格小于 1/lambda_max；0.85 留出收敛裕量
        alpha = 0.85 / lambda_max
        alpha = min(alpha, 0.1)

    if SCIPY_AVAILABLE:
        nodes = list(G.nodes())
        if not nodes:
            return {}
        A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, format="csr")
        I = sparse.identity(A.shape[0], format="csr", dtype=float)
        b = np.ones(A.shape[0], dtype=float)
        try:
            x = spsolve(I - alpha * A, b)
            x = np.asarray(x, dtype=float).reshape(-1)
            norm = np.linalg.norm(x)
            if norm > 0:
                x = x / norm
            return {n: float(v) for n, v in zip(nodes, x)}
        except Exception:
            pass

    try:
        return nx.katz_centrality(G, alpha=alpha, beta=1.0, max_iter=5000, tol=1e-8)
    except nx.PowerIterationFailedConvergence:
        # 再缩小一档 alpha 兜底
        alpha = alpha * 0.5
        return nx.katz_centrality(G, alpha=alpha, beta=1.0, max_iter=10000, tol=1e-8)



def compute_all_centralities(G: nx.Graph) -> pd.DataFrame:
    """对图中所有节点计算四种中心性。"""
    if G.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node_id", *CENTRALITY_COLUMNS])

    degree = nx.degree_centrality(G)
    katz = _compute_katz(G)
    closeness = nx.closeness_centrality(G)
    harmonic = nx.harmonic_centrality(G)

    rows = []
    for n in G.nodes():
        rows.append(
            {
                "node_id": n,
                "degree_centrality": float(degree.get(n, 0.0)),
                "katz_centrality": float(katz.get(n, 0.0)),
                "closeness_centrality": float(closeness.get(n, 0.0)),
                "harmonic_centrality": float(harmonic.get(n, 0.0)),
            }
        )
    return pd.DataFrame(rows)



def build_union_api_matrix(
    nodes_df: pd.DataFrame,
    centrality_df: pd.DataFrame,
    api_set_csv: str,
    aggregate: str = "max",
) -> pd.DataFrame:
    """
    按 API Set.csv 的 Union 列输出 426 x 4 矩阵。

    规则：
    1. 只用 external_api 节点与 API Set 匹配。
    2. 匹配键用 class_dot + 空格 + method_name_norm。
    3. 同一 API Set 条目匹配到多个重载节点时，默认取 max。
    """
    api_df = pd.read_csv(api_set_csv)
    if "Union" not in api_df.columns:
        raise ValueError("API Set.csv 缺少 Union 列")

    work = nodes_df.copy()
    work["node_id"] = work["node_id"].astype(str).str.strip()
    work["api_key"] = work.apply(lambda r: _safe_api_key(r.get("class_dot", ""), r.get("method_name_norm", "")), axis=1)

    work = work.merge(centrality_df, on="node_id", how="left")
    ext = work[work["owner_type"].astype(str).str.strip() == "external_api"].copy()

    targets = api_df["Union"].dropna().astype(str).str.strip().tolist()

    rows = []
    for api in targets:
        m = ext[ext["api_key"] == api]
        if m.empty:
            rows.append(
                {
                    "api": api,
                    "matched_nodes": 0,
                    "degree_centrality": 0.0,
                    "katz_centrality": 0.0,
                    "closeness_centrality": 0.0,
                    "harmonic_centrality": 0.0,
                }
            )
            continue

        if aggregate == "max":
            agg_row = {
                c: float(m[c].fillna(0.0).max())
                for c in CENTRALITY_COLUMNS
            }
        elif aggregate == "sum":
            agg_row = {
                c: float(m[c].fillna(0.0).sum())
                for c in CENTRALITY_COLUMNS
            }
        else:
            raise ValueError("aggregate 只支持 max 或 sum")

        rows.append(
            {
                "api": api,
                "matched_nodes": int(len(m)),
                **agg_row,
            }
        )

    return pd.DataFrame(rows)



def compute_and_save(
    graph_pkl: str,
    nodes_csv: str,
    api_set_csv: str,
    output_dir: str,
    aggregate: str = "max",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    with open(graph_pkl, "rb") as f:
        G = pickle.load(f)

    nodes_df = pd.read_csv(nodes_csv)
    all_cent = compute_all_centralities(G)
    all_cent.to_csv(os.path.join(output_dir, "all_nodes_centrality.csv"), index=False, encoding="utf-8-sig")

    union_df = build_union_api_matrix(
        nodes_df=nodes_df,
        centrality_df=all_cent,
        api_set_csv=api_set_csv,
        aggregate=aggregate,
    )
    union_df.to_csv(os.path.join(output_dir, "api_union_centrality_426x4.csv"), index=False, encoding="utf-8-sig")

    matrix = union_df[CENTRALITY_COLUMNS].to_numpy(dtype=float)
    pd.DataFrame(matrix, columns=CENTRALITY_COLUMNS).to_csv(
        os.path.join(output_dir, "api_union_matrix_only.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame(
        [
            {
                "num_graph_nodes": G.number_of_nodes(),
                "num_graph_edges": G.number_of_edges(),
                "num_union_rows": len(union_df),
                "num_union_apis_matched": int((union_df["matched_nodes"] > 0).sum()),
                "aggregate": aggregate,
            }
        ]
    )
    summary.to_csv(os.path.join(output_dir, "centrality_summary.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算所有节点四种中心性，并抽取 API Set 的 Union 矩阵")
    parser.add_argument("--graph_pkl", required=True)
    parser.add_argument("--nodes_csv", required=True)
    parser.add_argument("--api_set_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--aggregate", default="max", choices=["max", "sum"])
    args = parser.parse_args()

    compute_and_save(
        graph_pkl=args.graph_pkl,
        nodes_csv=args.nodes_csv,
        api_set_csv=args.api_set_csv,
        output_dir=args.output_dir,
        aggregate=args.aggregate,
    )
    print("中心性计算完成")
