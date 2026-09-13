"""链接预测离线评估：把一部分边当作「未来才产生的好友关系」隐藏起来，
检验推荐算法能否把它们排在候选前列。

运行方式（项目根目录下）：
    python -m backend.evaluate_link_prediction
"""

import os
import random
import tempfile

from .graph import Graph
from .storage import Storage
from .sample_data import generate_social_network
from . import link_prediction

HIDE_RATIO = 0.1  # 隐藏 10% 的边作为「未来好友关系」


def add_triadic_closure(nodes, edges, n_new, seed):
    """在 SBM 网络基础上模拟真实「好友的好友变成好友」的三元闭包过程，
    追加 n_new 条闭包边，使数据更接近真实社交网络（真实网络中共同好友
    越多越可能建立连接，纯 SBM 不具备这一性质）。
    """
    rng = random.Random(seed)
    adj = {}
    for e in edges:
        adj.setdefault(e["source"], set()).add(e["target"])
        adj.setdefault(e["target"], set()).add(e["source"])

    ids = [n["id"] for n in nodes]
    new_edges = list(edges)
    added, attempts = 0, 0
    while added < n_new and attempts < n_new * 50:
        attempts += 1
        u = rng.choice(ids)
        friends_u = adj.get(u)
        if not friends_u:
            continue
        w = rng.choice(sorted(friends_u))
        candidates = sorted(adj.get(w, ()) - friends_u - {u})
        if not candidates:
            continue
        v = rng.choice(candidates)
        adj[u].add(v)
        adj[v].add(u)
        new_edges.append({"source": u, "target": v, "weight": 1.0})
        added += 1
    return new_edges


def split_edges(edges, ratio, seed):
    """随机把边分成「当前可见」与「未来产生」两部分。"""
    rng = random.Random(seed)
    n_hide = int(len(edges) * ratio)
    hidden_idx = set(rng.sample(range(len(edges)), n_hide))
    remain = [e for i, e in enumerate(edges) if i not in hidden_idx]
    hidden = {
        tuple(sorted((e["source"], e["target"])))
        for i, e in enumerate(edges)
        if i in hidden_idx
    }
    return remain, hidden


def precision_at_k(graph, hidden, k, weights=None):
    """Top-K 推荐中命中隐藏边的比例。"""
    recs = link_prediction.recommend(graph, top_n=k, weights=weights)
    predicted = {tuple(sorted((r["source"], r["target"]))) for r in recs}
    return len(predicted & hidden) / k if k else 0.0


def main():
    nodes, edges = generate_social_network(seed=42)
    edges = add_triadic_closure(nodes, edges, n_new=int(len(edges) * 0.3), seed=13)
    remain, hidden = split_edges(edges, HIDE_RATIO, seed=7)

    with tempfile.TemporaryDirectory() as tmp:
        graph = Graph(Storage(os.path.join(tmp, "eval.db")))
        graph.import_data(nodes, remain)

        candidates = link_prediction.generate_candidates(graph)
        cand_set = {tuple(sorted(p)) for p in candidates}
        hidden_in_cand = hidden & cand_set
        random_baseline = len(hidden_in_cand) / len(cand_set) if cand_set else 0.0

        print(f"图规模：{graph.node_count()} 节点，可见边 {graph.edge_count()}，"
              f"隐藏边 {len(hidden)}")
        print(f"候选对（2 跳非好友）：{len(cand_set)}，"
              f"其中真实未来边 {len(hidden_in_cand)} 条，"
              f"随机基线 precision = {random_baseline:.4f}")
        print()

        # 完整模型与各单特征消融对比
        zero = {name: 0.0 for name in link_prediction.DEFAULT_WEIGHTS}
        models = [
            ("完整模型", None),
            ("仅共同好友", {**zero, "common_neighbors": 1.0}),
            ("仅 Adamic-Adar", {**zero, "adamic_adar": 1.0}),
            ("仅 Jaccard", {**zero, "jaccard": 1.0}),
            ("仅优先连接", {**zero, "preferential_attachment": 1.0}),
        ]
        header = f"{'模型':<14}" + "".join(f"P@{k:<8}" for k in (10, 50, 100))
        print(header)
        print("-" * len(header))
        for name, weights in models:
            row = f"{name:<14}"
            for k in (10, 50, 100):
                row += f"{precision_at_k(graph, hidden, k, weights):<10.3f}"
            print(row)
        print()
        print("（P@K = Top-K 推荐中命中「未来真实连边」的比例，越高越好）")


if __name__ == "__main__":
    main()
