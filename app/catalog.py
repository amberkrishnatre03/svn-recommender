"""
The product catalogue, kept in memory while the server runs.

Loaded once from Postgres when the server starts. Every request after that
reads from memory, which is thousands of times faster than asking the
database each time. After uploading new products, restart the server.
"""

import numpy as np

from app import database

print("loading catalogue...")
database.create_tables()
products, vectors = database.load_products()

if len(products) == 0:
    raise RuntimeError("no products in the database. Run: python build_embeddings.py")

# Every fashion product shares a big "generic clothing" direction, so everything
# looks similar to everything. Taking away the average leaves only what makes each
# product different, which makes the ranking much sharper.
vectors = vectors - vectors.mean(axis=0)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)   # back to length 1

# Columns copied out as plain arrays once, because reading a pandas table
# cell by cell is slow and we filter on these on every request.
gender_of = products["gender"].values
style_of = products["style"].values
category_of = products["category"].values
type_of = products["subcategory"].values
brand_of = products["brand"].values
is_trending = products["is_trending"].values.astype(bool)
is_sponsored = products["sponsored"].values.astype(bool)

# product_id -> row number, so finding a product is instant
row_of = {product_id: row for row, product_id in enumerate(products["product_id"])}

# What the app receives for each product, built once here instead of on every request
cards = []
for p in products.to_dict("records"):
    cards.append({
        "product_id": p["product_id"],
        "title": p["title"],
        "brand": p["brand"],
        "price": int(p["price"]) if p["price"] is not None else None,
        "image": p["primary_image"],
        "style": p["style"],
        "category": p["category"],
        "subcategory": p["subcategory"],
        "in_stock": bool(p["in_stock"]),
    })

all_styles = sorted(set(style_of))

print("loaded", len(products), "products")


def has_product(product_id):
    return product_id in row_of


def get_vector(product_id):
    return vectors[row_of[product_id]]


def get_card(row):
    return dict(cards[row])          # a copy, so adding fields later does not change the original