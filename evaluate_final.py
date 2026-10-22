"""Correct offline evaluation: remove test movies from train, then recommend."""
import csv, math
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\Lenovo\Doubao\chats\2026-10-20\new-chat\CineLab-Student-1.3.0-rc1\CineLab-Student-Windows-1.3.0-rc1")
PKG = ROOT / "data" / "packages" / "rec-course-v3-train5-1.0.0"
MODELS = ROOT / "models" / "business"

# Load test first
test = defaultdict(set)
with open(PKG / "evaluation" / "test.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        test[r["userId"]].add(r["movieId"])

# Load train, EXCLUDING test movies
train = defaultdict(dict)
with open(PKG / "business" / "history_ratings.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        u, m = r["userId"], r["movieId"]
        if m in test.get(u, set()):
            continue  # skip test movies from train
        train[u][m] = float(r["rating"])

# Only users with both train and test
users = [u for u in test if u in train and len(train[u]) > 0]
print(f"Evaluating on {len(users)} users")

# Load models
popular = []
with open(MODELS / "popular" / "popular_top.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        popular.append(r["movie_id"])

itemcf_sim = defaultdict(list)
with open(MODELS / "itemcf" / "itemcf_sim.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        itemcf_sim[r["movie_id"]].append((r["similar_id"], float(r["score"])))

content_sim = defaultdict(list)
with open(MODELS / "content" / "item_similarity.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        content_sim[r["movie_id"]].append((r["similar_id"], float(r["score"])))

als_factors = {}
with open(MODELS / "als" / "item_factors.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        vec = [float(v) for k,v in r.items() if k.startswith("g")]
        als_factors[r["movie_id"]] = vec

def rec_pop(u, n=10): return popular[:n]

def rec_itemcf(u, n=10):
    rated = train.get(u, {})
    scores = defaultdict(float)
    for mid, rt in rated.items():
        if rt < 3: continue
        for sid, ss in itemcf_sim.get(mid, []):
            if sid not in rated:
                scores[sid] += ss * rt
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [m for m,_ in ranked[:n]]

def rec_content(u, n=10):
    rated = train.get(u, {})
    scores = defaultdict(float)
    for mid, rt in rated.items():
        if rt < 3: continue
        w = rt - 3
        for sid, ss in content_sim.get(mid, []):
            if sid not in rated:
                scores[sid] += ss * w
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [m for m,_ in ranked[:n]]

def rec_als(u, n=10):
    rated = train.get(u, {})
    uvec = None; tw = 0
    for mid, rt in rated.items():
        if rt < 3: continue
        v = als_factors.get(mid)
        if not v: continue
        if uvec is None: uvec = [0.0]*len(v)
        for i in range(len(v)): uvec[i] += v[i]*rt
        tw += rt
    if not uvec or tw == 0: return popular[:n]
    un = math.sqrt(sum(x*x for x in uvec))
    if un > 0: uvec = [x/un for x in uvec]
    scores = [(mid, sum(a*b for a,b in zip(uvec, iv))) for mid,iv in als_factors.items() if mid not in rated]
    ranked = sorted(scores, key=lambda x:-x[1])
    return [m for m,_ in ranked[:n]]

algos = {"popular": rec_pop, "content": rec_content, "itemcf": rec_itemcf, "als": rec_als}

print(f"\n{'Algorithm':12s} {'P@10':>8s} {'R@10':>8s}")
print("-" * 32)
for name, fn in algos.items():
    ps, rs = [], []
    for u in users:
        truth = test[u]
        recs = set(fn(u, 10))
        hit = len(recs & truth)
        ps.append(hit/10.0)
        rs.append(hit/len(truth))
    print(f"{name:12s} {sum(ps)/len(ps):8.4f} {sum(rs)/len(rs):8.4f}")
