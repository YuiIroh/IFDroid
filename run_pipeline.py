from __future__ import annotations

import os

from build_graph import build_graph
from compute_centrality import compute_and_save


# =========================
# 这里硬编码输入/输出路径
# =========================
NODES_CSV = r"/mnt/data/nodes_intermediate.csv"
EDGES_CSV = r"/mnt/data/edges_intermediate.csv"
API_SET_CSV = r"/mnt/data/API Set.csv"
OUTPUT_DIR = r"/mnt/data/centrality_pipeline_output"
GRAPH_NAME = "call_graph.pkl"
USE_UNDIRECTED_GRAPH = True
API_AGGREGATE_MODE = "max"   # 可选: "max" 或 "sum"



def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/2] 开始建图...")
    build_graph(
        nodes_csv=NODES_CSV,
        edges_csv=EDGES_CSV,
        output_dir=OUTPUT_DIR,
        graph_name=GRAPH_NAME,
        undirected=USE_UNDIRECTED_GRAPH,
    )

    graph_pkl = os.path.join(OUTPUT_DIR, GRAPH_NAME)

    print("[2/2] 开始计算中心性...")
    compute_and_save(
        graph_pkl=graph_pkl,
        nodes_csv=NODES_CSV,
        api_set_csv=API_SET_CSV,
        output_dir=OUTPUT_DIR,
        aggregate=API_AGGREGATE_MODE,
    )

    print("全部完成。输出目录:", OUTPUT_DIR)
    print("- 图文件:", os.path.join(OUTPUT_DIR, GRAPH_NAME))
    print("- 所有节点中心性:", os.path.join(OUTPUT_DIR, "all_nodes_centrality.csv"))
    print("- API Set Union 中心性:", os.path.join(OUTPUT_DIR, "api_union_centrality_426x4.csv"))
    print("- 纯 426x4 数值矩阵:", os.path.join(OUTPUT_DIR, "api_union_matrix_only.csv"))


if __name__ == "__main__":
    main()
