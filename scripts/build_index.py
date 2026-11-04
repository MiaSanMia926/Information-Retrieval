import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from PIL import Image

print("--- 正在构建高质量索引 (V5) ---")

model = SentenceTransformer('clip-ViT-B-32')

def build_index(img_dir, index_path):
    img_list = []
    embeddings = []

    # 加载 captions.json
    captions = {}
    caption_path = ROOT / "data" / "captions.json"
    if os.path.exists(caption_path):
        try:
            with open(caption_path, "r", encoding="utf-8") as f:
                captions = json.load(f)
            print(f"✅ 已加载 captions.json，共 {len(captions)} 条描述")
        except Exception as e:
            print(f"⚠️ 加载 captions.json 失败: {e}")
    else:
        print("⚠️ 未找到 captions.json，将仅使用文件名")

    for file in sorted(os.listdir(img_dir)):
        if file.lower().endswith(('jpg', 'jpeg', 'png')):
            path = os.path.join(img_dir, file)
            try:
                img = Image.open(path).convert('RGB')
                emb = model.encode(img)
                
                embeddings.append(emb)
                img_list.append(Path(path).relative_to(ROOT).as_posix())
                
                # 输出时显示使用的文本
                caption = captions.get(file, "")
                text_used = caption if caption else file
                print(f"处理成功: {file} → {text_used[:60]}...")
                
            except Exception as e:
                print(f"失败: {file}, {e}")

    if not embeddings:
        raise ValueError("data/images 中没有可读取的 JPG/PNG 图片，请先准备图片数据。")

    embeddings = np.array(embeddings).astype('float32')

    # ✅ 关键改进1：归一化（变成 cosine similarity）
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    # ✅ 关键改进2：IVF 索引（比 Flat 高级）
    nlist = min(50, len(img_list))  # 聚类中心数（小数据可以调小）
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(embeddings)
    index.add(embeddings)

    # 搜索时探测更多簇
    index.nprobe = min(10, nlist)

    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "vector.index"))

    with open(os.path.join(index_path, "paths.txt"), "w", encoding="utf-8") as f:
        for p in img_list:
            f.write(p + "\n")

    print(f"✅ 完成：{len(img_list)} 张图片")

if __name__ == "__main__":
    build_index(ROOT / "data" / "images", ROOT / "data" / "index")