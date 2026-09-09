"""
Loads the products and their numbers into memory, one time, when the server starts till the time it is running.
"""

from pathlib import Path          # tool for building file paths

import numpy as np
import pandas as pd
                                                   # __file__ is a variable Python fills in automatically. It holds the path of the file currently running: /Users/ambertyagi/Documents/svn-recommender/app/catalog.py
DATA = Path(__file__).parent.parent / "data"        #path(...) turns text into path object we can navigate to / parent goes up one folder ( here twice)
                                              # /Users/ambertyagi/Documents/svn-recommender now we are here and the /data goes back into data folder
print("loading catalogue...")

products = pd.read_csv(DATA / "products_clean.csv")         # loads the products csv
vectors = np.load(DATA / "product_embeddings.npy")          # loads the embeddings numpy file

if len(products) != len(vectors):                            # each row is same and ordered of both products and vectors
    raise Exception("products and vectors do not match, rerun build_embeddings.py")          # rerun embeddings if not equal

# Every fashion product shares a big "generic clothing" direction, which makes
# everything look similar to everything else. Removing the average leaves only
# what makes each product different, so the ranking becomes much sharper. eg random products scored 0.42 similarity but after this its 0 and unrelated goes to negative
vectors = vectors - vectors.mean(axis=0)
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


row_of_product = {}                                                 # dicstionory to store the product id / table search 2.8 s and now its under 0.6 ms/ 5 thousand times fasster
for row_number, product_id in enumerate(products["product_id"]):            # builds a dictionray that maps product_id to its row number
    row_of_product[product_id] = row_number

# Reading single cells out of a pandas table is slow, and we do it thousands of
# times per request. Copying what we need into plain Python lists once at
# startup makes each feed about three times faster.
category_of = products["category"].tolist()           # copies two columns out of pandads table into oridinary lists  needed for recommended.py
subcategory_of = products["subcategory"].tolist()     # why ? because reading out of tables is slow reading out of lists is fast
brand_of = products["brand"].tolist()

ready_products = []
for p in products.to_dict("records"):          # to_dict turns the products pandas table inton dictionary , one per row with column names as keys
    ready_products.append({                    # PREBUILDING THE RESPONSES
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


def has_product(product_id):        # ASK WHETHER WE KNOW THIS PRODUCT
    return product_id in row_of_product             # A swipe can arrive for product not in our catalogue without this check_vector function would crash
                                                    # with key error and whole request would fail, callers check this first nd skip anything we dont recognise

def get_vector(product_id):          # looks up the rown number then take that row of numbersw.
    """The numbers for one product."""          # assumes the product exists which is why we have has products first
    return vectors[row_of_product[product_id]]


def get_product(row_number):               # returns one prebuilt dictionary
    """One product as a plain dictionary, ready to send to the app."""
    return dict(ready_products[row_number])            # dict(..) makes a copy 