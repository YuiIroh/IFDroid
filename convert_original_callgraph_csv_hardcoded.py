#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd

# =========================
# 硬编码输入输出路径：只改这里
# =========================
NODES_CSV = r"E:\Text to image\IFDroid\out_callgraphs\base/nodes.csv"
EDGES_CSV = r"E:\Text to image\IFDroid\out_callgraphs\base/edges.csv"
API_SET_CSV = None   # 不想匹配 API Set 时，可改成 None
OUT_DIR = r"E:\Text to image\IFDroid\out_callgraphs\baseconverted_callgraph_hardcoded"

PRIMITIVE_MAP = {
    'V': 'void',
    'Z': 'boolean',
    'B': 'byte',
    'S': 'short',
    'C': 'char',
    'I': 'int',
    'J': 'long',
    'F': 'float',
    'D': 'double',
}


def to_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {'1', 'true', 'yes', 'y', 't'}


def dex_type_to_java(dex_name: str) -> str:
    if dex_name is None:
        return ''
    dex_name = str(dex_name).strip()
    if not dex_name:
        return ''

    arr = 0
    while dex_name.startswith('['):
        arr += 1
        dex_name = dex_name[1:]

    if dex_name in PRIMITIVE_MAP:
        base = PRIMITIVE_MAP[dex_name]
    elif dex_name.startswith('L') and dex_name.endswith(';'):
        base = dex_name[1:-1].replace('/', '.')
    else:
        base = dex_name.replace('/', '.')

    return base + '[]' * arr


def compact_descriptor(desc: str) -> str:
    if desc is None:
        return ''
    return re.sub(r'\s+', '', str(desc).strip())


def parse_signature_text(sig: str):
    """
    兼容原 build_callgraphs.py 导出的 signature，例如：
      Landroid/os/Environment; getExternalStorageDirectory ()Ljava/io/File;
      La/a; a (F F F I)Landroid/window/BackEvent;
    """
    sig = '' if sig is None else str(sig).strip()
    if not sig:
        return {
            'class_dex': '',
            'method_name_norm': '',
            'descriptor_raw': '',
            'descriptor_compact': '',
        }

    m = re.match(r'^(L[^;]+;)\s+(\S+)\s+(.+)$', sig)
    if m:
        cls_dex = m.group(1).strip()
        method = m.group(2).strip()
        desc_raw = m.group(3).strip()
        return {
            'class_dex': cls_dex,
            'method_name_norm': method,
            'descriptor_raw': desc_raw,
            'descriptor_compact': compact_descriptor(desc_raw),
        }

    m = re.match(r'^(L[^;]+;)->([^\(\s]+)(.+)$', sig)
    if m:
        cls_dex = m.group(1).strip()
        method = m.group(2).strip()
        desc_raw = m.group(3).strip()
        return {
            'class_dex': cls_dex,
            'method_name_norm': method,
            'descriptor_raw': desc_raw,
            'descriptor_compact': compact_descriptor(desc_raw),
        }

    return {
        'class_dex': '',
        'method_name_norm': '',
        'descriptor_raw': '',
        'descriptor_compact': '',
    }


def is_probably_obfuscated(class_dot: str, method_name: str, external_flag: bool) -> bool:
    if external_flag:
        return False
    if not class_dot or not method_name:
        return False
    class_tail = class_dot.split('.')[-1]
    short_class = bool(re.fullmatch(r'[A-Za-z]{1,2}', class_tail))
    short_method = bool(re.fullmatch(r'[A-Za-z]{1,2}|<init>|<clinit>', method_name))
    return short_class or short_method


def normalize_node_row(row) -> dict:
    signature = row.get('signature', '')
    parsed = parse_signature_text(signature)

    class_dex = parsed['class_dex'] or str(row.get('classname', '') or row.get('class_name', '')).strip()
    method_name = parsed['method_name_norm'] or str(row.get('methodname', '') or row.get('method_name', '')).strip()
    descriptor_raw = parsed['descriptor_raw'] or str(row.get('descriptor', '')).strip()
    descriptor_compact = parsed['descriptor_compact'] or compact_descriptor(descriptor_raw)

    external_flag = to_bool(row.get('external', False))
    class_dot = dex_type_to_java(class_dex)
    api_text = f'{class_dot} {method_name}'.strip() if class_dot and method_name else ''
    method_key = f'{class_dot}->{method_name}{descriptor_compact}'.strip() if class_dot and method_name else ''
    owner_type = 'external_api' if external_flag else 'app_method'

    out = dict(row)
    out.update({
        'class_dex': class_dex,
        'class_dot': class_dot,
        'method_name_norm': method_name,
        'descriptor_raw': descriptor_raw,
        'descriptor_compact': descriptor_compact,
        'api_text': api_text,
        'method_key': method_key,
        'owner_type': owner_type,
        'is_obfuscated_suspect': is_probably_obfuscated(class_dot, method_name, external_flag),
    })
    return out


def build_node_lookup(df_nodes_norm: pd.DataFrame):
    cols = [
        'signature', 'class_dex', 'class_dot', 'method_name_norm',
        'descriptor_raw', 'descriptor_compact', 'api_text', 'method_key',
        'owner_type', 'external', 'is_obfuscated_suspect'
    ]
    cols = [c for c in cols if c in df_nodes_norm.columns]
    lookup = {}
    for _, row in df_nodes_norm.iterrows():
        sig = str(row['signature'])
        lookup[sig] = {c: row[c] for c in cols}
    return lookup


def attach_edge_metadata(df_edges: pd.DataFrame, node_lookup: dict) -> pd.DataFrame:
    rows = []
    for _, row in df_edges.iterrows():
        rec = dict(row)
        src = str(row.get('src', ''))
        dst = str(row.get('dst', ''))
        src_meta = node_lookup.get(src, {})
        dst_meta = node_lookup.get(dst, {})
        rec.update({
            'src_class_dot': src_meta.get('class_dot', ''),
            'src_method_name_norm': src_meta.get('method_name_norm', ''),
            'src_descriptor_compact': src_meta.get('descriptor_compact', ''),
            'src_api_text': src_meta.get('api_text', ''),
            'src_method_key': src_meta.get('method_key', ''),
            'src_owner_type': src_meta.get('owner_type', ''),
            'dst_class_dot': dst_meta.get('class_dot', ''),
            'dst_method_name_norm': dst_meta.get('method_name_norm', ''),
            'dst_descriptor_compact': dst_meta.get('descriptor_compact', ''),
            'dst_api_text': dst_meta.get('api_text', ''),
            'dst_method_key': dst_meta.get('method_key', ''),
            'dst_owner_type': dst_meta.get('owner_type', ''),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def load_api_union(api_set_path: Path):
    df_api = pd.read_csv(api_set_path)
    if 'Union' not in df_api.columns:
        raise ValueError('API Set.csv 中没有 Union 列。')
    union = df_api['Union'].dropna().astype(str).str.strip()
    union = union[union != '']
    union_unique = sorted(set(union.tolist()))
    return df_api, union_unique


def add_api_matches(df_nodes_norm: pd.DataFrame, df_edges_norm: pd.DataFrame, union_unique):
    union_set = set(union_unique)

    df_nodes_norm = df_nodes_norm.copy()
    df_nodes_norm['match_union'] = df_nodes_norm['api_text'].isin(union_set)

    df_edges_norm = df_edges_norm.copy()
    df_edges_norm['src_match_union'] = df_edges_norm['src_api_text'].isin(union_set)
    df_edges_norm['dst_match_union'] = df_edges_norm['dst_api_text'].isin(union_set)

    matched_nodes = df_nodes_norm[df_nodes_norm['match_union']].copy()
    presence = pd.DataFrame({'Union': union_unique})
    present_set = set(matched_nodes['api_text'].dropna().astype(str))
    presence['present_in_sample'] = presence['Union'].isin(present_set)
    presence['matched_node_rows'] = presence['Union'].map(
        matched_nodes['api_text'].value_counts().to_dict()
    ).fillna(0).astype(int)

    return df_nodes_norm, df_edges_norm, matched_nodes, presence


def main():
    nodes_path = Path(NODES_CSV)
    edges_path = Path(EDGES_CSV)
    api_set_path = Path(API_SET_CSV) if API_SET_CSV else None
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not nodes_path.exists():
        raise FileNotFoundError(f'nodes.csv 不存在: {nodes_path}')
    if not edges_path.exists():
        raise FileNotFoundError(f'edges.csv 不存在: {edges_path}')
    if api_set_path is not None and not api_set_path.exists():
        raise FileNotFoundError(f'API Set.csv 不存在: {api_set_path}')

    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)

    required_node_cols = {'signature'}
    required_edge_cols = {'src', 'dst'}
    if not required_node_cols.issubset(df_nodes.columns):
        raise ValueError(f'nodes.csv 缺少必要列: {sorted(required_node_cols - set(df_nodes.columns))}')
    if not required_edge_cols.issubset(df_edges.columns):
        raise ValueError(f'edges.csv 缺少必要列: {sorted(required_edge_cols - set(df_edges.columns))}')

    nodes_norm_records = [normalize_node_row(row) for _, row in df_nodes.iterrows()]
    df_nodes_norm = pd.DataFrame(nodes_norm_records)

    preferred_front = [
        'node_id', 'signature', 'class_dex', 'class_dot', 'method_name_norm',
        'descriptor_raw', 'descriptor_compact', 'api_text', 'method_key',
        'owner_type', 'external', 'entrypoint', 'is_obfuscated_suspect',
        'classname', 'methodname', 'accessflags',
    ]
    preferred_front = [c for c in preferred_front if c in df_nodes_norm.columns]
    other_cols = [c for c in df_nodes_norm.columns if c not in preferred_front]
    df_nodes_norm = df_nodes_norm[preferred_front + other_cols]

    node_lookup = build_node_lookup(df_nodes_norm)
    df_edges_norm = attach_edge_metadata(df_edges, node_lookup)

    df_nodes_norm.to_csv(out_dir / 'nodes_intermediate.csv', index=False, encoding='utf-8-sig')
    df_edges_norm.to_csv(out_dir / 'edges_intermediate.csv', index=False, encoding='utf-8-sig')

    summary = {
        'input_nodes_csv': str(nodes_path),
        'input_edges_csv': str(edges_path),
        'input_api_set_csv': str(api_set_path) if api_set_path else None,
        'output_dir': str(out_dir),
        'total_nodes': int(len(df_nodes_norm)),
        'total_edges': int(len(df_edges_norm)),
        'external_nodes': int(df_nodes_norm['external'].map(to_bool).sum()) if 'external' in df_nodes_norm.columns else None,
        'app_method_nodes': int((df_nodes_norm['owner_type'] == 'app_method').sum()),
        'external_api_nodes': int((df_nodes_norm['owner_type'] == 'external_api').sum()),
        'obfuscated_suspect_nodes': int(df_nodes_norm['is_obfuscated_suspect'].sum()),
    }

    if api_set_path is not None:
        _, union_unique = load_api_union(api_set_path)
        df_nodes_norm2, df_edges_norm2, matched_nodes, presence = add_api_matches(df_nodes_norm, df_edges_norm, union_unique)

        df_nodes_norm2.to_csv(out_dir / 'nodes_intermediate_with_match.csv', index=False, encoding='utf-8-sig')
        df_edges_norm2.to_csv(out_dir / 'edges_intermediate_with_match.csv', index=False, encoding='utf-8-sig')
        matched_nodes.to_csv(out_dir / 'matched_union_nodes.csv', index=False, encoding='utf-8-sig')
        presence.to_csv(out_dir / 'union_presence_426.csv', index=False, encoding='utf-8-sig')

        summary.update({
            'api_set_union_size': int(len(union_unique)),
            'matched_node_rows': int(len(matched_nodes)),
            'matched_unique_api_text': int(matched_nodes['api_text'].nunique()),
            'matched_external_nodes': int(len(matched_nodes[matched_nodes['owner_type'] == 'external_api'])),
        })

    pd.DataFrame([summary]).to_csv(out_dir / 'match_summary.csv', index=False, encoding='utf-8-sig')
    with open(out_dir / 'match_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'输出目录: {out_dir}')


if __name__ == '__main__':
    main()
