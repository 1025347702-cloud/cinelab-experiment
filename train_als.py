"""Train real ALS matrix factorization on evaluation/train.csv."""
import csv
import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from pathlib import Path

ROOT = Path(r"C:\Users\Lenovo\Doubao\chats\2026-10-20\new-chat\CineLab-Student-1.3.0-rc1\CineLab-Student-Windows-1.3.0-rc1")
PKG = ROOT / "data" / "packages" / "rec-course-v3-train5-1.0.0"
OUT = ROOT / "models" / "business" / "als"

# Load train.csv
print("Loading train.csv...")
rows = []
with open(PKG / "evaluation" / "train.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append((int(r["userId"]), int(r["movieId"]), float(r["rating"])))
print(f"Rows: {len(rows)}")

# Build user-item matrix
user_ids = sorted(set(r[0] for r in rows))
movie_ids = sorted(set(r[1] for r in rows))
user_map = {u: i for i, u in enumerate(user_ids)}
movie_map = {m: i for i, m in enumerate(movie_ids)}

# Build user x item matrix
data, r_idx, c_idx = [], [], []
for u, m, rt in rows:
    r_idx.append(user_map[u])
    c_idx.append(movie_map[m])
    data.append(rt)

# user x item
user_item = csr_matrix((data, (r_idx, c_idx)), shape=(len(user_ids), len(movie_ids)))
print(f"user_item matrix: {user_item.shape} (users x items)")

# Train ALS: implicit expects item x user matrix
item_user = user_item.T.tocsr()  # shape: (n_items, n_users)
print(f"item_user matrix: {item_user.shape} (items x users)")

model = AlternatingLeastSquares(
    factors=20,
    iterations=10,
    regularization=0.1,
    random_state=42
)
model.fit(item_user)
print("Training done")

print(f"model.item_factors shape: {model.item_factors.shape}")
print(f"model.user_factors shape: {model.user_factors.shape}")

# implicit 0.7: when input is item x user matrix:
#   user_factors -> item factors (8459, 20)
#   item_factors -> user factors (4500, 20)
item_factors = model.user_factors
n_items = item_factors.shape[0]
print(f"Using movie factors: {n_items} x {item_factors.shape[1]}")

# Export
out_path = OUT / "item_factors.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    header = ["movie_id"] + [f"g{i+1}" for i in range(20)]
    w.writerow(header)
    for i in range(n_items):
        w.writerow([movie_ids[i]] + [round(float(x), 6) for x in item_factors[i]])

print(f"Saved {n_items} movies to {out_path}")
