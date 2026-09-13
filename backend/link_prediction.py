"""好友推荐（链接预测）：找出「该认识却还没认识」的节点对，并给出理由。

整体思路
--------
1. 候选生成：只考察「好友的好友」（2 跳可达）且当前不是好友的节点对。
   避免 O(V^2) 全量配对，也符合真实社交产品「二度人脉」的推荐场景。
2. 特征：共同好友数、Jaccard、Adamic-Adar、优先连接、是否同社群
   （Louvain 划分）、资料属性重合度。
3. 打分：连续特征在候选集内做 log1p + min-max 归一后加权求和，
   权重可通过 weights 参数覆盖。
4. 解释：按各特征对总分的贡献排序，取贡献最高的若干项生成中文理由。

所有计算直接读取内存邻接表（Graph.adj），不访问数据库。
"""

import math

# 默认特征权重，按对样例数据的留出评估（backend/evaluate_link_prediction.py）
# 调优；生产环境应改用历史「加好友」数据训练 LR/GBDT 学习权重。
# preferential_attachment 默认置 0：
# 它只反映节点热度，容易让「大 V」霸榜，对「该认识」的刻画没有增量。
DEFAULT_WEIGHTS = {
    "adamic_adar": 0.45,
    "common_neighbors": 0.20,
    "jaccard": 0.15,
    "same_community": 0.10,
    "attr_similarity": 0.10,
    "preferential_attachment": 0.0,
}

# 需要归一化的连续特征（same_community 本身是 0/1，直接加权）
_CONTINUOUS = (
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "attr_similarity",
)


def generate_candidates(graph, node=None):
    """生成候选节点对：2 跳内可达但尚不是好友的配对。

    node 为 None 时返回全图候选（无向去重），
    否则只返回包含该节点的候选，用于「你可能认识」单人推荐。
    时间复杂度约为 O(Σ deg(w)^2)，w 为中间节点。
    """
    pairs = set()
    seeds = [str(node)] if node is not None else graph.node_ids()
    for u in seeds:
        friends_u = set(graph.adj.get(u, {}))
        for w in friends_u:
            for v in graph.adj.get(w, {}):
                if v == u or v in friends_u:
                    continue
                pairs.add((u, v) if u < v else (v, u))
    return sorted(pairs)


def pair_features(graph, u, v, community=None):
    """计算一对节点的链接预测特征，附带解释所需的原始信息。"""
    adj = graph.adj
    fu, fv = set(adj.get(u, {})), set(adj.get(v, {}))
    common = fu & fv
    union = fu | fv

    # Adamic-Adar：共同好友的度数越小，这条纽带越「稀有」，信号越强。
    # 共同好友的度数至少为 2（连着 u 和 v），log 不会除零。
    adamic_adar = sum(1.0 / math.log(max(graph.degree(w), 2)) for w in common)

    same_community = 0
    if community is not None and u in community and v in community:
        same_community = int(community[u] == community[v])

    # 资料属性重合：(key, value) 完全一致的属性，如同城、同公司、同标签
    attrs_u = graph.nodes.get(u, {}).get("attrs") or {}
    attrs_v = graph.nodes.get(v, {}).get("attrs") or {}
    shared_attrs = sorted(
        f"{k}={attrs_u[k]}"
        for k in set(attrs_u) & set(attrs_v)
        if attrs_u[k] == attrs_v[k]
    )
    n_keys = len(set(attrs_u) | set(attrs_v))

    return {
        "common_neighbors": len(common),
        "jaccard": len(common) / len(union) if union else 0.0,
        "adamic_adar": adamic_adar,
        "preferential_attachment": len(fu) * len(fv),
        "same_community": same_community,
        "attr_similarity": len(shared_attrs) / n_keys if n_keys else 0.0,
        # 以下两项不参与打分，仅用于生成解释
        "mutual_friends": sorted(common),
        "shared_attrs": shared_attrs,
    }


def recommend(graph, node=None, top_n=10, weights=None):
    """计算好友推荐列表，按得分降序返回 top_n 条。

    每条结果：{source, target, score, reasons, features, mutual_friends}
    """
    weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
    community = graph.communities()  # 惰性计算并缓存 Louvain 划分

    candidates = generate_candidates(graph, node)
    if not candidates:
        return []

    feats = {p: pair_features(graph, p[0], p[1], community) for p in candidates}

    # 连续特征在候选集内 log1p（压缩长尾）+ min-max 归一
    norm = {name: {} for name in _CONTINUOUS}
    for name in _CONTINUOUS:
        vals = {p: math.log1p(feats[p][name]) for p in candidates}
        lo, hi = min(vals.values()), max(vals.values())
        span = hi - lo
        for p in candidates:
            norm[name][p] = (vals[p] - lo) / span if span > 0 else 0.0

    results = []
    for p in candidates:
        contrib = {
            name: weights[name] * norm[name][p] for name in _CONTINUOUS
        }
        contrib["same_community"] = weights["same_community"] * feats[p]["same_community"]

        results.append(
            {
                "source": p[0],
                "target": p[1],
                "score": round(sum(contrib.values()), 4),
                "reasons": _explain(graph, feats[p], contrib),
                "features": {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in feats[p].items()
                    if k not in ("mutual_friends", "shared_attrs")
                },
                "mutual_friends": feats[p]["mutual_friends"],
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_n]


def _explain(graph, feats, contrib):
    """按特征贡献生成中文理由，取贡献最高的 3 项。"""
    top = sorted(contrib.items(), key=lambda kv: kv[1], reverse=True)
    reasons = []
    for name, value in top:
        if value <= 0 or len(reasons) >= 3:
            continue
        reasons.append(_render_reason(graph, name, feats))
    if not reasons:
        reasons.append("是好友的好友，存在二度人脉连接")
    return reasons


def _render_reason(graph, name, feats):
    if name == "common_neighbors":
        friends = feats["mutual_friends"]
        shown = "、".join(friends[:5])
        suffix = " 等" if len(friends) > 5 else ""
        return f"有 {len(friends)} 位共同好友（{shown}{suffix}）"
    if name == "jaccard":
        return f"好友圈重合度高（Jaccard {feats['jaccard']:.2f}）"
    if name == "adamic_adar":
        # 找出度数最小的共同好友：圈子越小，纽带越稀有、越能说明问题
        rare = sorted(feats["mutual_friends"], key=graph.degree)[:2]
        return f"共同好友 {'、'.join(rare)} 的社交圈很小，属于强纽带信号"
    if name == "preferential_attachment":
        return "两人都是活跃节点，产生连接的先验概率高"
    if name == "attr_similarity":
        return f"资料标签一致：{'、'.join(feats['shared_attrs'][:3])}"
    if name == "same_community":
        return "同属一个社群"
    return ""
