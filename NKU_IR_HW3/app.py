import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from rank_bm25 import BM25Okapi
import os

st.set_page_config(page_title="多模态检索系统 Pro+", layout="wide")
st.title("🚀 多模态智能检索系统")

# =========================
# 🔧 加载资源
# =========================
@st.cache_resource
def load_all():
    model = SentenceTransformer('clip-ViT-B-32')
    index = faiss.read_index("index/vector.index")

    with open("index/paths.txt") as f:
        img_paths = [l.strip() for l in f.readlines()]

    # 👉 BM25：用文件名作为“文本语料”
    docs = [os.path.basename(p) for p in img_paths]
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    return model, index, img_paths, bm25, docs

model, index, img_paths, bm25, docs = load_all()

# =========================
# 🧠 用户反馈存储（简单版）
# =========================
if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # {(query, path): +1/-1}

# =========================
# ⚙️ Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ 检索策略")

    use_query_expansion = st.checkbox("查询扩展（多Prompt）", True)
    use_bm25 = st.checkbox("BM25融合", True)
    use_feedback = st.checkbox("用户反馈优化", True)

    uploaded_file = st.file_uploader("📂 上传图片", type=['jpg','png'])

# =========================
# 🔍 输入
# =========================
query = st.text_input("请输入查询（英文效果更佳）")

# =========================
# 🧠 查询编码（含扩展）
# =========================
def encode_query(query):
    if use_query_expansion:
        prompts = [
            f"a photo of {query}",
            f"a detailed image of {query}",
            f"{query} in real world"
        ]
        embs = model.encode(prompts)
        q = np.mean(embs, axis=0)
    else:
        q = model.encode([query])[0]

    q = q.astype('float32')
    faiss.normalize_L2(q.reshape(1, -1))
    return q

# =========================
# 🚀 检索逻辑
# =========================
if uploaded_file:
    query_img = Image.open(uploaded_file).convert('RGB')
    q = model.encode(query_img).astype('float32')
    q = q.reshape(1, -1)
    faiss.normalize_L2(q)

    is_text_query = False

elif query:
    q = encode_query(query).reshape(1, -1)
    is_text_query = True
else:
    q = None

if q is not None:
    with st.spinner("🔍 检索中..."):

        # ========= 第一阶段：CLIP召回 =========
        D, I = index.search(q, k=20)

        candidates = []
        clip_scores = []

        for i, idx in enumerate(I[0]):
            candidates.append(img_paths[idx])
            clip_scores.append(D[0][i])

        # ========= BM25 =========
        if is_text_query and use_bm25:
            token_q = query.lower().split()
            bm25_scores = bm25.get_scores(token_q)

            bm25_dict = {
                img_paths[i]: bm25_scores[i]
                for i in range(len(img_paths))
            }
        else:
            bm25_dict = {}

        # ========= 融合 =========
        final_scores = []

        for i, path in enumerate(candidates):
            score = clip_scores[i]

            # 👉 BM25融合
            if path in bm25_dict:
                score += 0.3 * bm25_dict[path]

            # 👉 用户反馈调整
            if use_feedback:
                key = (query, path)
                if key in st.session_state.feedback:
                    score += 0.1 * st.session_state.feedback[key]

            final_scores.append((path, score))

        # ========= 排序 =========
        final_results = sorted(final_scores, key=lambda x: x[1], reverse=True)[:6]

    # =========================
    # 🖼️ 展示结果
    # =========================
    st.subheader("🔍 检索结果")

    cols = st.columns(3)

    for i, (path, score) in enumerate(final_results):
        with cols[i % 3]:
            st.image(path, use_container_width=True)
            st.caption(f"综合得分: {score:.4f}")

            # 👍 👎 反馈按钮
            if st.button("👍 相关", key=f"pos_{i}"):
                st.session_state.feedback[(query, path)] = 1

            if st.button("👎 不相关", key=f"neg_{i}"):
                st.session_state.feedback[(query, path)] = -1