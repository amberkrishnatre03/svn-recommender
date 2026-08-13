"""
Loads the products and their numbers into memory, one time, when the server starts.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data"

print("loading catalogue...")

products = pd.read_csv(DATA / "products_clean.csv")
vectors = np.load(DATA / "product_embeddings.npy")

if len(products) != len(vectors):
    raise Exception("products and vectors do not match, rerun build_embeddings.py")

# Every fashion product shares a big "generic clothing" direction, which makes
# everything look similar to everything else. Removing the average leaves only
# what makes each product different, so the ranking becomes much sharper.
vectors = vectors - vectors.mean(axis=0)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

row_of_product = {}
for row_number, product_id in enumerate(products["product_id"]):
    row_of_product[product_id] = row_number

# Reading single cells out of a pandas table is slow, and we do it thousands of
# times per request. Copying what we need into plain Python lists once at
# startup makes each feed about three times faster.
category_of = products["category"].tolist()
subcategory_of = products["subcategory"].tolist()

ready_products = []
for p in products.to_dict("records"):
    ready_products.append({
        "product_id": p["product_id"],
        "title": p["title"],
        "brand": p["brand"],
        "price": int(p["price"]),
        "image": p["primary_image"],
        "style": p["style"],
        "category": p["category"],
        "subcategory": p["subcategory"],
        "in_stock": bool(p["in_stock"]),
    })

print("loaded", len(products), "products")


def get_vector(product_id):
    """The numbers for one product."""
    return vectors[row_of_product[product_id]]


def get_product(row_number):
    """One product as a plain dictionary, ready to send to the app."""
    return dict(ready_products[row_number])