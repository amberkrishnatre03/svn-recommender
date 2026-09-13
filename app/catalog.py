"""
The product catalogue, kept in memory while the server runs.

Every product is stored under its product_id, together with its own vector:
    products["SNITCH_859..."] = {"title": ..., "style": ..., "vector": 512 numbers}

Loaded once from Postgres when the server starts. Every request after that
reads from memory, which is much faster than asking the database each time.
After new products are stored, restart the server so it loads them.

Nothing here is hardcoded. Styles, categories, subcategories and brands are
whatever the database happens to contain, and neighbour styles are worked out
from the products themselves.
"""

import numpy as np

from app import database

print("loading catalogue...")
database.create_tables()
all_products = database.load_products()          # a list of dictionaries, one per product

if len(all_products) == 0:
    raise RuntimeError("no products in the database. Run: python build_embeddings.py")

# Every fashion product shares a big "generic clothing" direction, so everything
# looks similar to everything. Taking away the average vector leaves only what
# makes each product different, which makes the matching much sharper.
all_vectors = []
for product in all_products:
    all_vectors.append(product["vector"])       # collect every product's 512 numbers
average = np.mean(all_vectors, axis=0)          # the average of each of the 512 numbers

products = {}                                    # product_id -> the product, with its vector
styles = set()                                   # every style name that exists
for product in all_products:
    vector = product["vector"] - average
    product["vector"] = vector / np.linalg.norm(vector)     # back to length 1
    products[product["product_id"]] = product
    styles.add(product["style"])

all_styles = sorted(styles)


# ---------------------------------------------------------- neighbour styles
# Which styles sit next to each other used to be a dictionary of style names
# written by hand. That was a bug waiting to happen: adding a style to the
# database gave it no neighbours at all, silently, so a user who picked it saw
# only that one style.
#
# Instead we work it out from the catalogue. Each style's average product is its
# "centre", and the styles whose centres are most alike are its neighbours.
# Whatever styles the database holds, this gives every one of them neighbours.

NEIGHBOURS_PER_STYLE = 2        # how many nearby styles a picked style pulls in

style_centre = {}
for style in all_styles:
    of_this_style = [p["vector"] for p in products.values() if p["style"] == style]
    mean = np.mean(of_this_style, axis=0)
    length = np.linalg.norm(mean)
    style_centre[style] = mean / length if length > 1e-9 else mean

neighbours = {}
for style in all_styles:
    others = [s for s in all_styles if s != style]
    others.sort(key=lambda other: float(np.dot(style_centre[style], style_centre[other])),
                reverse=True)
    neighbours[style] = others[:NEIGHBOURS_PER_STYLE]

print("loaded", len(products), "products")
print("styles:", ", ".join(all_styles))
for style in all_styles:
    print(f"   {style} -> {neighbours[style]}")


def has_product(product_id):
    return product_id in products


def get_vector(product_id):
    return products[product_id]["vector"]


def get_card(product_id):
    """What the app receives for one product: every detail except the vector and embedding_text.
    A value missing in the database is None, which the app receives as null."""
    product = products[product_id]
    card = {}
    for column in product:
        if column == "vector" or column == "embedding_text":    # only for the model
            continue
        card[column] = product[column]
    card["image"] = product["primary_image"]                    # the app already reads "image"
    return card