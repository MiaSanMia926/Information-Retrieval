# 多模态图像检索 · Multimodal Image Retrieval

基于 CLIP 与 FAISS 的本地图像检索应用，支持文本搜图、以图搜图，以及可切换的 BM25 融合与会话内相关性反馈。使用 Streamlit 展示检索结果，适合观察语义召回和关键词重排的差异。

## 功能与实现

| 能力 | 实现 |
| --- | --- |
| 文本 / 图片查询 | `sentence-transformers` 的 `clip-ViT-B-32` 统一编码 |
| 向量召回 | L2 归一化向量 + FAISS IVF 内积检索 |
| 文本查询扩展 | 三种英文提示词编码后取均值 |
| 混合排序 | CLIP 分数叠加文件名 BM25 分数 |
| 相关性反馈 | 当前 Streamlit 会话内对结果加减分 |
| 离线评估 | P@5、R@5、MRR@5，比较 CLIP 与两种 BM25 权重 |

## 项目结构

```text
.
├── app.py                  # Streamlit 入口
├── scripts/
│   ├── build_index.py      # 图片编码与 FAISS 索引构建
│   └── evaluate.py         # 离线检索评估
├── data/
│   ├── captions.json       # 可编辑的图片描述，键为图片文件名
│   ├── images/             # 本地图片，不提交到 Git
│   └── index/              # 生成的 vector.index、paths.txt
├── docs/                   # 设计、开发与评估说明
│   └── archive/            # 原始设计报告和历史索引快照
└── requirements.txt
```

## 快速开始

建议使用 Python 3.11，在仓库根目录执行：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

1. 将有权使用的 JPG / JPEG / PNG 图片放入 `data/images/`。原始仓库未包含图片数据，需要自行准备。
2. 构建当前数据对应的索引：`python scripts/build_index.py`。
3. 启动应用：`python -m streamlit run app.py`。
4. 打开终端显示的本地地址（默认 `http://localhost:8501`），输入英文描述或上传图片查询。

首次编码会下载模型，需要网络连接。替换、删除或增加图片后应重新构建索引。脚本按源码位置定位数据，不再依赖旧的作业目录名称。

## 文档

- [系统设计与排序逻辑](docs/architecture.md)
- [开发与数据约定](docs/development.md)
- [评估口径与限制](docs/evaluation.md)
- [目录迁移记录](docs/migration.md)
- [贡献指南](CONTRIBUTING.md)

当前是单机应用：反馈不跨会话保存；评估脚本与页面的 BM25 语料、融合方式不同，不能把脚本分数直接当作页面效果。仓库不声明未经复现的检索精度或性能数据。

本次目录整理的实际检查结果见[验证记录](docs/validation.md)。
