# -*- coding: utf-8 -*-

"""
硬编码版：
批量 APK -> Androguard 分析 -> 导出函数调用图

固定输入目录：
E:\Text to image\APK

固定输出目录：
E:\Text to image\out_callgraphs

运行方式：
python build_callgraphs.py
"""

import json
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import pandas as pd
from tqdm import tqdm
from androguard.misc import AnalyzeAPK


# =============================
# 硬编码路径
# =============================
APK_DIR = Path(r"E:\Text to image\APK")
OUT_DIR = Path(r"E:\Text to image\IFDroid\out_callgraphs")


# =============================
# 工具函数
# =============================
def safe_call(obj: Any, method_name: str, default=None):
    fn = getattr(obj, method_name, None)
    if callable(fn):
        try:
            return fn()
        except Exception:
            return default
    return default


def sanitize_value(v: Any):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple, set)):
        return json.dumps(list(v), ensure_ascii=False)
    if isinstance(v, dict):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def split_signature(sig: str) -> Tuple[str, str, str]:
    """
    把方法签名拆成:
    Lpkg/Class;->method(desc)return
    """
    m = re.match(r"^(L[^;]+;)->([^(]+)(\(.*\).*)$", sig)
    if not m:
        return "", "", ""
    return m.group(1), m.group(2), m.group(3)


def node_to_signature(node: Any) -> str:
    """
    尽量把节点对象转成稳定字符串签名：
    Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;
    """
    full_name = getattr(node, "full_name", None)
    if isinstance(full_name, str) and full_name.strip():
        return full_name.strip()

    method_obj = getattr(node, "method", None)
    if method_obj is not None:
        cls = safe_call(method_obj, "get_class_name")
        name = safe_call(method_obj, "get_name")
        desc = safe_call(method_obj, "get_descriptor")
        if cls and name and desc:
            return f"{cls}->{name}{desc}"

        full_name2 = getattr(method_obj, "full_name", None)
        if isinstance(full_name2, str) and full_name2.strip():
            return full_name2.strip()

    get_method = getattr(node, "get_method", None)
    if callable(get_method):
        try:
            m = get_method()
            cls = safe_call(m, "get_class_name")
            name = safe_call(m, "get_name")
            desc = safe_call(m, "get_descriptor")
            if cls and name and desc:
                return f"{cls}->{name}{desc}"

            full_name3 = getattr(m, "full_name", None)
            if isinstance(full_name3, str) and full_name3.strip():
                return full_name3.strip()
        except Exception:
            pass

    if isinstance(node, str):
        return node.strip()

    return str(node).strip()


def pack_node_attrs(sig: str, raw_attrs: Dict[str, Any]) -> Dict[str, Any]:
    class_name, method_name, descriptor = split_signature(sig)
    out = {
        "signature": sig,
        "class_name": class_name,
        "method_name": method_name,
        "descriptor": descriptor,
    }
    for k, v in raw_attrs.items():
        out[k] = sanitize_value(v)
    return out


def multidigraph_to_digraph(g_multi: nx.MultiDiGraph) -> nx.DiGraph:
    """
    把 MultiDiGraph 压成更方便导出的 DiGraph：
    - 节点改成字符串签名
    - 多重边合并为一条边
    - 保留 call_count / offsets
    """
    g = nx.DiGraph()

    sig_map = {}
    for node, attrs in g_multi.nodes(data=True):
        sig = node_to_signature(node)
        sig_map[node] = sig
        merged_attrs = pack_node_attrs(sig, attrs)

        if not g.has_node(sig):
            g.add_node(sig, **merged_attrs)
        else:
            for k, v in merged_attrs.items():
                if k not in g.nodes[sig] or g.nodes[sig][k] in ("", None):
                    g.nodes[sig][k] = v

    for u, v, attrs in g_multi.edges(data=True):
        su = sig_map[u]
        sv = sig_map[v]
        offset = attrs.get("offset", None)

        if g.has_edge(su, sv):
            g[su][sv]["call_count"] += 1
            offsets = json.loads(g[su][sv]["offsets"])
            offsets.append(offset)
            g[su][sv]["offsets"] = json.dumps(offsets, ensure_ascii=False)
        else:
            g.add_edge(
                su,
                sv,
                call_count=1,
                offsets=json.dumps([offset], ensure_ascii=False),
            )

    return g


def export_graph_outputs(g: nx.DiGraph, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # graph.gml
    nx.write_gml(g, out_dir / "graph.gml")

    # nodes.csv
    node_rows = []
    for node, attrs in g.nodes(data=True):
        row = {"node_id": node}
        row.update(attrs)
        node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(
        out_dir / "nodes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # edges.csv
    edge_rows = []
    for u, v, attrs in g.edges(data=True):
        row = {"src": u, "dst": v}
        row.update(attrs)
        edge_rows.append(row)
    pd.DataFrame(edge_rows).to_csv(
        out_dir / "edges.csv",
        index=False,
        encoding="utf-8-sig"
    )


def analyze_one_apk(apk_path: Path, out_root: Path) -> Dict[str, Any]:
    sample_name = apk_path.stem
    sample_out = out_root / sample_name
    sample_out.mkdir(parents=True, exist_ok=True)

    result = {
        "apk_name": apk_path.name,
        "apk_path": str(apk_path),
        "ok": False,
        "package_name": "",
        "version_name": "",
        "version_code": "",
        "min_sdk": "",
        "target_sdk": "",
        "permissions_declared": 0,
        "nodes": 0,
        "edges": 0,
        "external_nodes": 0,
        "internal_nodes": 0,
        "error": "",
    }

    try:
        a, d, dx = AnalyzeAPK(str(apk_path))

        meta = {
            "apk_file": str(apk_path),
            "package_name": a.get_package(),
            "app_name": safe_call(a, "get_app_name"),
            "version_name": a.get_androidversion_name(),
            "version_code": a.get_androidversion_code(),
            "min_sdk": a.get_min_sdk_version(),
            "target_sdk": a.get_target_sdk_version(),
            "max_sdk": a.get_max_sdk_version(),
            "permissions": a.get_permissions(),
            "activities": safe_call(a, "get_activities", []),
            "services": safe_call(a, "get_services", []),
            "receivers": safe_call(a, "get_receivers", []),
            "providers": safe_call(a, "get_providers", []),
        }

        with open(sample_out / "apk_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 获取调用图
        g_multi = dx.get_call_graph(no_isolated=True)

        # 转成普通 DiGraph
        g = multidigraph_to_digraph(g_multi)

        # 导出
        export_graph_outputs(g, sample_out)

        external_nodes = 0
        for _, attrs in g.nodes(data=True):
            ext = attrs.get("external", False)
            if isinstance(ext, str):
                ext = ext.lower() == "true"
            if ext:
                external_nodes += 1

        result.update({
            "ok": True,
            "package_name": meta["package_name"],
            "version_name": meta["version_name"],
            "version_code": meta["version_code"],
            "min_sdk": meta["min_sdk"],
            "target_sdk": meta["target_sdk"],
            "permissions_declared": len(meta["permissions"]),
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "external_nodes": external_nodes,
            "internal_nodes": g.number_of_nodes() - external_nodes,
        })

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        with open(sample_out / "error.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())

    return result


def main():
    print("=" * 60)
    print(f"APK_DIR : {APK_DIR}")
    print(f"OUT_DIR : {OUT_DIR}")
    print("=" * 60)

    if not APK_DIR.exists():
        print(f"[错误] APK 目录不存在: {APK_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    apk_files = sorted(APK_DIR.rglob("*.apk"))
    if not apk_files:
        print(f"[错误] 在目录中没有找到 APK 文件: {APK_DIR}")
        return

    print(f"[信息] 共找到 {len(apk_files)} 个 APK")

    all_rows: List[Dict[str, Any]] = []
    for apk_path in tqdm(apk_files, desc="Analyzing APKs"):
        row = analyze_one_apk(apk_path, OUT_DIR)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    ok_cnt = int(df["ok"].sum()) if "ok" in df.columns else 0
    print("=" * 60)
    print(f"Total APKs : {len(apk_files)}")
    print(f"Success    : {ok_cnt}")
    print(f"Failed     : {len(apk_files) - ok_cnt}")
    print(f"Summary    : {OUT_DIR / 'summary.csv'}")
    print("=" * 60)


if __name__ == "__main__":
    main()