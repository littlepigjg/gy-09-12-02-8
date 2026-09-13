# 社交网络分析服务

一个前后端分离的社交网络分析服务：导入用户关系数据构建图结构，支持最短路径查询、共同好友、PageRank 排名与 Louvain 社群发现，前端以 Vue + Cytoscape 可视化网络图、路径高亮与社群着色。

## 功能特性

- **图数据导入**：支持导入节点/边 JSON 数据，或一键生成带社群结构的样例网络
- **最短路径**：BFS 广度优先搜索，前端高亮展示路径
- **共同好友**：基于邻接表集合求交集
- **好友推荐（链接预测）**：对 2 跳非好友对计算共同好友 / Jaccard / Adamic-Adar / 社群 / 资料属性等特征加权得分，输出 Top-N「该认识却还没认识」的配对并生成中文理由
- **PageRank**：简化幂迭代实现，输出影响力 Top-N 排名
- **社群发现**：Louvain 算法（局部移动 + 社区聚合），前端按社群着色
- **图统计**：节点数、边数、平均度、密度等基础指标

## 技术栈

| 层 | 技术 |
|----|------|
| 后端 | Python 3.12 + Flask |
| 存储 | SQLite（节点表 / 边表） |
| 前端 | Vue 3 + Cytoscape.js（本地化，无需外网） |

## 项目结构

```
.
├── backend/
│   ├── app.py           # Flask REST API
│   ├── storage.py       # SQLite 节点/边表持久化
│   ├── graph.py         # 内存邻接表缓存
│   ├── algorithms.py    # BFS / 共同好友 / PageRank / Louvain
│   ├── link_prediction.py        # 好友推荐（链接预测）与解释生成
│   ├── evaluate_link_prediction.py # 链接预测留出评估（precision@K）
│   └── sample_data.py   # 随机块模型样例数据生成
├── frontend/
│   ├── index.html
│   ├── app.js           # Vue 3 + Cytoscape 应用
│   ├── style.css
│   └── vendor/          # 本地化的 vue/cytoscape 库
├── Dockerfile
├── docker-compose.yml
├── start-docker.sh      # Docker 一键启动
├── run.sh               # 本地运行
└── requirements.txt
```

## 快速开始

### 方式一：Docker 一键启动（推荐）

```bash
# 构建镜像并启动（默认端口 8000）
bash start-docker.sh

# 或指定端口
bash start-docker.sh 8080
```

启动后访问 http://localhost:8000 ，点击「加载样例网络」即可看到网络图。

其他命令：

```bash
docker compose up -d --build   # 构建并后台启动
docker compose logs -f         # 查看日志
docker compose down            # 停止并删除容器
```

数据持久化：SQLite 数据库挂载在 `./data` 目录，删除容器后数据仍保留。

### 方式二：本地运行

```bash
pip install -r requirements.txt
bash run.sh 8000
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/import` | 导入关系数据 `{nodes, edges}` |
| POST | `/api/load_sample` | 生成并导入内置样例网络 |
| GET  | `/api/graph` | 获取全部节点与边 |
| GET  | `/api/stats` | 图基础统计 |
| GET  | `/api/shortest_path?from&to` | BFS 最短路径 |
| GET  | `/api/common_friends?node1&node2` | 共同好友 |
| GET  | `/api/friend_recommendations?node=&top_n=` | 好友推荐：不带 node 返回全图 Top 配对，带 node 返回该用户的「你可能认识」 |
| GET  | `/api/pagerank?top_n=10` | PageRank 排名 |
| GET  | `/api/communities` | Louvain 社群划分 |
| GET  | `/api/neighbors?node` | 节点邻居 |

### 导入数据格式

```json
{
  "nodes": [
    {"id": "1", "name": "Alice"},
    {"id": "2", "name": "Bob"}
  ],
  "edges": [
    {"source": "1", "target": "2", "weight": 1.0}
  ]
}
```

若省略 `nodes`，后端会从 `edges` 的 source/target 自动推断节点。

## 算法说明

- **BFS 最短路径**：无权图广度优先搜索，时间复杂度 O(V+E)
- **共同好友**：两节点邻接集合求交集，O(min(deg(a), deg(b)))
- **好友推荐（链接预测）**：
  - 候选生成：只考察「好友的好友」（2 跳可达）的非好友对，避免 O(V²) 全量配对
  - 特征：共同好友数、Jaccard、Adamic-Adar（共同好友度数越小权重越高）、优先连接、是否同社群（Louvain）、资料属性重合度
  - 打分：连续特征在候选集内 log1p + min-max 归一后加权求和，默认权重见 `link_prediction.DEFAULT_WEIGHTS`；优先连接默认置 0 以避免「大 V」霸榜
  - 解释：按各特征对总分的贡献取 Top-3，渲染为中文理由（共同好友名单、强纽带、同社群、资料标签一致等）
  - 离线评估：`python -m backend.evaluate_link_prediction` 隐藏 10% 边作为「未来好友关系」，输出 precision@K 与各单特征消融对比（样例数据上 P@10 约为随机基线的 10 倍）；生产环境应改用历史加好友数据训练 LR/GBDT 学习权重
- **PageRank（简化）**：幂迭代法，阻尼系数 0.85，按出度均分权重，悬挂节点权重回流
- **Louvain 社群发现**：两阶段迭代——局部移动（按模块度增益 `ΔQ = k_i,in/m - Σ_tot·k_i/(2m²)` 移动节点）+ 社区聚合（将社区收缩为超节点递归），多层级执行至模块度收敛

## 存储与优化设计

针对大规模图的三个核心难点：

1. **存储与查询优化**：节点表/边表分离持久化到 SQLite，边表对 source/target 建索引；查询统一走内存邻接表缓存，避免频繁回库。
2. **算法内存高效执行**：邻接表以 `dict[str, dict[str, float]]` 常驻内存，导入后仅 O(V+E) 构建一次，所有遍历均为内存级 O(V+E) 访问。
3. **可视化交互性能**：Cytoscape 画布渲染 + `haystack` 边 + 力导向布局；路径高亮、社群着色均通过 class 切换，避免全量重建元素。
