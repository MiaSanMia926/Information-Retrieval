import numpy as np
import faiss
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "index"
CAPTIONS_PATH = ROOT / "data" / "captions.json"
import json
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

print("=== 多模态检索系统评估 (V4 - 支持 Caption) ===\n")

model = SentenceTransformer('clip-ViT-B-32')
index = faiss.read_index(str(INDEX_DIR / "vector.index"))

with (INDEX_DIR / "paths.txt").open(encoding="utf-8") as f:
    img_paths = [line.strip() for line in f.readlines()]

# 加载 captions
captions = {}
if CAPTIONS_PATH.exists():
    with CAPTIONS_PATH.open(encoding="utf-8") as f:
        captions = json.load(f)
    print(f"✅ 已加载 {len(captions)} 条图像描述")
else:
    print("⚠️ 未找到 captions.json")

# 构建 BM25 语料（优先使用 caption）
docs = []
for path in img_paths:
    basename = os.path.basename(path)
    caption = captions.get(basename, "")
    if caption:
        doc = caption.lower()
    else:
        doc = basename.replace('.jpg', '').replace('.png', '').replace('_', ' ').lower()
    docs.append(doc)

bm25 = BM25Okapi([d.split() for d in docs])

print(f"加载完成：{len(img_paths)} 张图片\n")

# =========================
# 测试集（已适配你的文件名）
# =========================
GROUND_TRUTH = {
    "bread": ["fresh_bread.jpg"],
    "pancakes": ["pancakes_with_blueberries.jpg"],
    "coffee": ["coffee_latte_art.jpg"],
    "dog": ["cute_brown_puppy.jpg", "black_dog_portrait.jpg", "golden_puppy.jpg"],
    "puppy": ["cute_brown_puppy.jpg", "golden_puppy.jpg"],
    "cat": ["cat_in_cherry_blossoms.jpg", "small_tabby_kitten.jpg"],
    "animals": ["dogs_and_cats_group.jpg"],
    "fox": ["red_fox_closeup.jpg"],
    "deer": ["young_deer_in_forest.jpg"],
    "snake": ["yellow_snake_coiled.jpg"],
    "zebra": ["zebra_in_grassland.jpg"],
    "elephant": ["elephant(animal).jpg"],
    "rhino": ["rhino(animal).jpg"],
    "chicken": ["chickens_in_snow.jpg"],
    "flower": ["blue_cosmos_flower.jpg", "red_yellow_tulip.jpg"],
    "mushroom": ["red_fly_agaric_mushroom.jpg"],
    "tree": ["lonely_tree_in_field.jpg", "green_tree_canopy.jpg"],
    "sunset": ["birds_at_sunset.jpg"],
    "landscape": ["abandoned_boat_in_desert.jpg", "mountain_lake_landscape.jpg"],
    "mountain": ["hill.jpg"],
    "snow": ["snow.jpg"],
    "woman": ["woman_in_white_dress.jpg"],
    "girl": ["young_girl_in_red.jpg"],
    "boy": ["smiling_boy_in_field.jpg"],
    "family": ["father_and_daughter_winter.jpg"],
    "piano": ["grand_piano_keyboard.jpg"],
    "chess": ["chess_pieces.jpg"],
    "building": ["building.jpg"],
    "road": ["road.jpg", "pedestrian_crosswalk.jpg"],
    "presentation": ["mathematics_presentation_slide.jpg", "course_planning_ppt.jpg", "three_color_planning.jpg", "week7_task_overview.jpg"],
}

def evaluate_system(use_bm25=False, bm25_weight=0.15, use_feedback=False):
    precision_list, recall_list, mrr_list = [], [], []
    
    for query, true_files in GROUND_TRUTH.items():
        q_vec = model.encode([query]).astype('float32')
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, k=len(img_paths))
        
        scores = []
        for rank, idx in enumerate(I[0]):
            if idx < 0:
                continue
            path = img_paths[idx]
            score = 1.0 / (rank + 60)
            
            if use_bm25:
                basename = os.path.basename(path)
                caption = captions.get(basename, "")
                text = caption if caption else basename.replace('.jpg','').replace('.png','').replace('_',' ')
                try:
                    bm25_scores = bm25.get_scores(query.lower().split())
                    bm25_score = bm25_scores[docs.index(text.lower())]
                    score += bm25_weight * (bm25_score / (max(bm25_scores) + 1e-5))
                except:
                    pass  # 如果找不到就跳过
            
            if use_feedback:
                score += 0.08
                
            scores.append((path, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        top5 = [os.path.basename(p) for p, _ in scores[:5]]
        
        true_set = set(true_files)
        hit = len(true_set & set(top5))
        
        precision_list.append(hit / 5)
        recall_list.append(hit / len(true_set) if true_set else 0)
        
        mrr = 0
        for r, fname in enumerate(top5, 1):
            if fname in true_set:
                mrr = 1.0 / r
                break
        mrr_list.append(mrr)
    
    return {
        "P@5": np.mean(precision_list),
        "R@5": np.mean(recall_list),
        "MRR": np.mean(mrr_list)
    }

print("正在进行 Ablation Study...\n")

print("1. Pure CLIP")
r1 = evaluate_system(use_bm25=False)
print(f"   P@5: {r1['P@5']:.1%}   R@5: {r1['R@5']:.1%}   MRR: {r1['MRR']:.3f}")

print("\n2. CLIP + BM25 (weight=0.3)")
r2 = evaluate_system(use_bm25=True, bm25_weight=0.3)
print(f"   P@5: {r2['P@5']:.1%}   R@5: {r2['R@5']:.1%}   MRR: {r2['MRR']:.3f}")

print("\n3. CLIP + BM25 (weight=0.15)")
r3 = evaluate_system(use_bm25=True, bm25_weight=0.15)
print(f"   P@5: {r3['P@5']:.1%}   R@5: {r3['R@5']:.1%}   MRR: {r3['MRR']:.3f}")

print("\n✅ 评估完成！请把结果复制到报告中。")