#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量将 Androguard 导出的 nodes.csv / edges.csv 标准化为 API Set 风格命名。

功能：
1. 扫描 INPUT_ROOT 下每个 APK 子文件夹。
2. 读取其中的 nodes.csv 和 edges.csv。
3. 修复 class_name/method_name 为空时，自动回退到 classname/methodname。
4. 生成简化版 nodes_api_named.csv 和 edges_api_named.csv。
5. 用 api_text = class_dot + " " + method_name_norm 与 API Set 的 Union 列做精确匹配。
6. 可选导出 graph_api_named.gml / graph_api_named.json / summary.json。

说明：
- 这个脚本是“中心性计算前的预处理”。
- 四种中心性仍然需要基于完整图（nodes + edges）来计算。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional, Dict

import networkx as nx
import pandas as pd


# =========================
# 硬编码路径（按你的实际路径修改）
# =========================
INPUT_ROOT = r"E:\Text to image\APK_graphs_input"   # 里面有多个 APK 文件夹
OUTPUT_ROOT = r"E:\Text to image\APK_graphs_output"
API_SET_CSV = r"E:\Text to image\API Set.csv"

# 可选输出
WRITE_GML = True
WRITE_JSON = True
COPY_APK_META_JSON = True
GRAPH_UNDIRECTED = True


# =========================
# 基础映射
# =========================
PRIMITIVE_MAP = {
    "V": "void",
    "Z": "boolean",
    "B": "byte",
    "S": "short",
    "C": "char",
    "I": "int",
    "J": "long",
    "F": "float",
    "D": "double",
}


# =========================
# 通用辅助函数
# =========================
def safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def to_bool(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in {"1", "true", "t", "yes", "y"}


def pick_best_series(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    """优先用 primary，若为空则回退到 fallback。"""
    if primary in df.columns:
        a = df[primary]
    else:
        a = pd.Series([""] * len(df), index=df.index)

    if fallback in df.columns:
        b = df[fallback]
    else:
        b = pd.Series([""] * len(df), index=df.index)

    a_ok = a.notna() & (a.astype(str).str.strip() != "")
    return a.where(a_ok, b)


# =========================
# 类型 / 描述符处理
# =========================
def dex_type_to_human(desc: str) -> str:
    if not desc:
        return ""

    dims = 0
    while desc.startswith("["):
        dims += 1
        desc = desc[1:]

    if desc in PRIMITIVE_MAP:
        base = PRIMITIVE_MAP[desc]
    elif desc.startswith("L") and desc.endswith(";"):
        base = desc[1:-1].replace("/", ".")
    else:
        base = desc

    return base + "[]" * dims


def dex_class_to_dot(name: str) -> str:
    s = safe_str(name)
    if not s:
        return ""
    if s.startswith("["):
        return dex_type_to_human(s)
    if s.startswith("L") and s.endswith(";"):
        return s[1:-1].replace("/", ".")
    return s.replace("/", ".")


def normalize_descriptor_compact(desc) -> str:
    s = safe_str(desc)
    if not s or "(" not in s or ")" not in s:
        return s

    left = s.find("(")
    right = s.find(")")
    args_part = s[left + 1:right]
    ret_part = s[right + 1:]

    args = []
    i = 0
    while i < len(args_part):
        ch = args_part[i]
        if ch == "[":
            j = i
            while j < len(args_part) and args_part[j] == "[":
                j += 1
            if j < len(args_part) and args_part[j] == "L":
                k = args_part.find(";", j)
                if k == -1:
                    token = args_part[i:]
                    i = len(args_part)
                else:
                    token = args_part[i:k + 1]
                    i = k + 1
            else:
                token = args_part[i:j + 1]
                i = j + 1
            args.append(dex_type_to_human(token))
        elif ch == "L":
            k = args_part.find(";", i)
            if k == -1:
                token = args_part[i:]
                i = len(args_part)
            else:
                token = args_part[i:k + 1]
                i = k + 1
            args.append(dex_type_to_human(token))
        else:
            args.append(dex_type_to_human(ch))
            i += 1

    ret_human = dex_type_to_human(ret_part)
    return f"({','.join(args)})->{ret_human}"


# =========================
# API Set 读取
# =========================
def load_api_union(api_set_csv: str) -> List[str]:
    df = pd.read_csv(api_set_csv)
    if "Union" not in df.columns:
        raise ValueError(f"API Set 文件中未找到 'Union' 列: {api_set_csv}")
    union = df["Union"].dropna().astype(str).str.strip()
    union = union[union != ""].tolist()
    return union


# =========================
# 节点标准化
# =========================
def normalize_nodes(nodes_df: pd.DataFrame, api_union_set: set) -> pd.DataFrame:
    df = nodes_df.copy()

    # 关键修复：优先 class_name/method_name，空时回退到 classname/methodname
    df["class_raw"] = pick_best_series(df, "class_name", "classname").apply(safe_str)
    df["method_raw"] = pick_best_series(df, "method_name", "methodname").apply(safe_str)

    # 描述符也做兼容
    if "descriptor" in df.columns:
        df["descriptor_raw"] = df["descriptor"].apply(safe_str)
    else:
        df["descriptor_raw"] = ""

    df["class_dot"] = df["class_raw"].apply(dex_class_to_dot)
    df["method_name_norm"] = df["method_raw"]
    df["descriptor_compact"] = df["descriptor_raw"].apply(normalize_descriptor_compact)

    # external / entrypoint 兼容处理
    if "external" in df.columns:
        df["external_bool"] = df["external"].apply(to_bool)
    else:
        df["external_bool"] = False

    if "entrypoint" in df.columns:
        df["entrypoint_bool"] = df["entrypoint"].apply(to_bool)
    else:
        df["entrypoint_bool"] = False

    df["owner_type"] = df["external_bool"].map(lambda x: "external_api" if x else "app_method")

    # API Set 对齐键
    df["api_text"] = (
        df["class_dot"].fillna("").astype(str).str.strip()
        + " "
        + df["method_name_norm"].fillna("").astype(str).str.strip()
    ).str.strip()

    # 若后面你要区分重载，可以用这个更精确的键
    df["method_key"] = (
        df["class_dot"].fillna("").astype(str).str.strip()
        + " "
        + df["method_name_norm"].fillna("").astype(str).str.strip()
        + " "
        + df["descriptor_compact"].fillna("").astype(str).str.strip()
    ).str.strip()

    df["api_union_exact_match"] = df["api_text"].isin(api_union_set)
    df["matched_api_name"] = df["api_text"].where(df["api_union_exact_match"], "")

    # 简化输出：只保留关键信息
    keep_cols = [
        "node_id",
        "class_dot",
        "method_name_norm",
        "descriptor_compact",
        "api_text",
        "method_key",
        "owner_type",
        "external_bool",
        "entrypoint_bool",
        "api_union_exact_match",
        "matched_api_name",
    ]

    # 确保 node_id 存在
    if "node_id" not in df.columns:
        raise ValueError("nodes.csv 中缺少 node_id 列，无法构图。")

    return df[keep_cols].copy()


# =========================
# 边标准化
# =========================
def normalize_edges(edges_df: pd.DataFrame, nodes_norm: pd.DataFrame) -> pd.DataFrame:
    df = edges_df.copy()

    if "src" not in df.columns or "dst" not in df.columns:
        raise ValueError("edges.csv 中缺少 src 或 dst 列，无法构图。")

    node_meta = nodes_norm[[
        "node_id",
        "api_text",
        "owner_type",
        "api_union_exact_match",
    ]].copy()

    src_meta = node_meta.rename(columns={
        "node_id": "src",
        "api_text": "src_api_text",
        "owner_type": "src_owner_type",
        "api_union_exact_match": "src_api_union_exact_match",
    })

    dst_meta = node_meta.rename(columns={
        "node_id": "dst",
        "api_text": "dst_api_text",
        "owner_type": "dst_owner_type",
        "api_union_exact_match": "dst_api_union_exact_match",
    })

    df = df.merge(src_meta, on="src", how="left")
    df = df.merge(dst_meta, on="dst", how="left")

    if "call_count" not in df.columns:
        df["call_count"] = 1

    keep_cols = [
        "src",
        "dst",
        "call_count",
        "src_api_text",
        "dst_api_text",
        "src_owner_type",
        "dst_owner_type",
        "src_api_union_exact_match",
        "dst_api_union_exact_match",
    ]
    return df[keep_cols].copy()


# =========================
# 构图与导出
# =========================
def build_graph_from_normalized(nodes_norm: pd.DataFrame, edges_norm: pd.DataFrame):
    G = nx.Graph() if GRAPH_UNDIRECTED else nx.DiGraph()

    for row in nodes_norm.itertuples(index=False):
        G.add_node(
            row.node_id,
            api_text=getattr(row, "api_text", ""),
            method_key=getattr(row, "method_key", ""),
            owner_type=getattr(row, "owner_type", ""),
            api_union_exact_match=bool(getattr(row, "api_union_exact_match", False)),
        )

    for row in edges_norm.itertuples(index=False):
        src = getattr(row, "src")
        dst = getattr(row, "dst")
        if pd.isna(src) or pd.isna(dst):
            continue
        call_count = getattr(row, "call_count", 1)
        if pd.isna(call_count):
            call_count = 1
        G.add_edge(src, dst, call_count=int(call_count))

    return G


def write_graph_json(G, out_json: Path) -> None:
    data = nx.node_link_data(G)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# 单个样本处理
# =========================
def process_one_sample(sample_dir: Path, out_root: Path, api_union: List[str]) -> Optional[Dict[str, object]]:
    nodes_csv = sample_dir / "nodes.csv"
    edges_csv = sample_dir / "edges.csv"

    if not nodes_csv.exists() or not edges_csv.exists():
        return None

    out_dir = out_root / sample_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_df = pd.read_csv(nodes_csv)
    edges_df = pd.read_csv(edges_csv)
    api_union_set = set(api_union)

    nodes_norm = normalize_nodes(nodes_df, api_union_set)
    edges_norm = normalize_edges(edges_df, nodes_norm)

    nodes_out = out_dir / "nodes_api_named.csv"
    edges_out = out_dir / "edges_api_named.csv"
    nodes_norm.to_csv(nodes_out, index=False, encoding="utf-8-sig")
    edges_norm.to_csv(edges_out, index=False, encoding="utf-8-sig")

    G = build_graph_from_normalized(nodes_norm, edges_norm)

    if WRITE_GML:
        nx.write_gml(G, out_dir / "graph_api_named.gml")

    if WRITE_JSON:
        write_graph_json(G, out_dir / "graph_api_named.json")

    if COPY_APK_META_JSON:
        meta_src = sample_dir / "apk_meta.json"
        if meta_src.exists():
            shutil.copy2(meta_src, out_dir / "apk_meta.json")

    matched_api_names = (
        nodes_norm.loc[nodes_norm["api_union_exact_match"], "api_text"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    summary = {
        "apk_folder": sample_dir.name,
        "nodes_total": int(len(nodes_norm)),
        "edges_total": int(len(edges_norm)),
        "external_nodes": int((nodes_norm["owner_type"] == "external_api").sum()),
        "app_method_nodes": int((nodes_norm["owner_type"] == "app_method").sum()),
        "matched_sensitive_nodes": int(nodes_norm["api_union_exact_match"].sum()),
        "matched_sensitive_api_count": int(len(set(matched_api_names))),
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "graph_type": "undirected" if GRAPH_UNDIRECTED else "directed",
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    pd.DataFrame({"matched_api_name": sorted(set(matched_api_names))}).to_csv(
        out_dir / "matched_api_union_names.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return summary


# =========================
# 主流程
# =========================
def main():
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    api_union = load_api_union(API_SET_CSV)

    sample_dirs = [p for p in input_root.iterdir() if p.is_dir()]
    sample_dirs.sort(key=lambda p: p.name.lower())

    all_rows = []
    for sample_dir in sample_dirs:
        try:
            row = process_one_sample(sample_dir, output_root, api_union)
            if row is None:
                print(f"[SKIP] {sample_dir.name}: 缺少 nodes.csv 或 edges.csv")
            else:
                all_rows.append(row)
                print(f"[OK] {sample_dir.name}")
        except Exception as e:
            print(f"[ERROR] {sample_dir.name}: {e}")

    if all_rows:
        pd.DataFrame(all_rows).to_csv(output_root / "batch_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\nDone. Summary saved to: {output_root / 'batch_summary.csv'}")
    else:
        print("\nNo valid sample folders were processed.")


if __name__ == "__main__":
    main()
