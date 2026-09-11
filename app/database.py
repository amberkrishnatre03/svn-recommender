"""
Everything that talks to Postgres.

Two tables:
    user_taste   one row per user: gender, taste vectors, seen products, styles
    products     one row per product: its details and its 512 numbers

The address of the database comes from the DATABASE_URL environment variable,
so the same code works on your Mac (.env file) and on the cloud (cloud settings).
"""

import os

import numpy as np
import pandas as pd
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

VECTOR_SIZE = 512       # every FashionCLIP vector has 512 numbers
SEEN_LIMIT = 5000       # we remember at most this many shown products per user

# The product columns we store, same names as in the products CSV
PRODUCT_COLUMNS = ["product_id", "title", "brand", "gender", "category", "subcategory",
                   "style", "price", "in_stock", "is_active", "primary_image",
                   "is_trending", "sponsored", "embedding_text"]

# A small set of open connections that every request shares, instead of
# opening a new connection each time (which is slow).
pool = ConnectionPool(
    os.getenv("DATABASE_URL"),
    min_size=1,
    max_size=20,
    check=ConnectionPool.check_connection,   # test a connection before handing it out
    max_idle=300,                            # close connections unused for 5 minutes
    max_lifetime=1800,                       # replace every connection after 30 minutes
    open=True,
)


def create_tables():
    """Make both tables if they do not exist yet. Safe to run every time."""
    with pool.connection() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS user_taste (
                user_id   TEXT PRIMARY KEY,
                gender    TEXT NOT NULL,
                vector    FLOAT8[] NOT NULL,
                seen_ids  TEXT[] NOT NULL DEFAULT '{}',
                styles    TEXT[] NOT NULL DEFAULT '{}'
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id      TEXT PRIMARY KEY,
                title           TEXT,
                brand           TEXT,
                gender          TEXT NOT NULL,
                category        TEXT,
                subcategory     TEXT,
                style           TEXT NOT NULL,
                price           INTEGER,
                in_stock        BOOLEAN,
                is_active       BOOLEAN NOT NULL DEFAULT TRUE,
                primary_image   TEXT,
                is_trending     BOOLEAN NOT NULL DEFAULT FALSE,
                sponsored       BOOLEAN NOT NULL DEFAULT FALSE,
                embedding_text  TEXT,
                embedding       REAL[] NOT NULL
            )
        """)


# ---------- users ----------

def save_new_user(user_id, gender, tastes, styles):
    """Create the user, or start them completely fresh if they already exist."""
    numbers = tastes.flatten().tolist()          # (2, 512) table -> one list of 1,024 numbers
    with pool.connection() as db:
        db.execute("""
            INSERT INTO user_taste (user_id, gender, vector, seen_ids, styles)
            VALUES (%s, %s, %s, '{}', %s)
            ON CONFLICT (user_id) DO UPDATE SET
                gender = EXCLUDED.gender,
                vector = EXCLUDED.vector,
                seen_ids = '{}',
                styles = EXCLUDED.styles
        """, (user_id, gender, numbers, styles))


def get_user(user_id):
    """Returns (gender, tastes, seen_ids), or None if we have never seen this user."""
    with pool.connection() as db:
        row = db.execute(
            "SELECT gender, vector, seen_ids FROM user_taste WHERE user_id = %s",
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    gender, numbers, seen_ids = row
    tastes = np.array(numbers, dtype="float32").reshape(-1, VECTOR_SIZE)   # back into rows of 512
    return gender, tastes, list(seen_ids)


def update_user(user_id, tastes, seen_ids):
    """Save the new taste and the list of products they have seen."""
    numbers = tastes.flatten().tolist()
    seen_ids = seen_ids[-SEEN_LIMIT:]            # keep only the most recent ones
    with pool.connection() as db:
        db.execute(
            "UPDATE user_taste SET vector = %s, seen_ids = %s WHERE user_id = %s",
            (numbers, seen_ids, user_id),
        )


def clear_seen(user_id):
    """Forget which products they have seen, keep their taste."""
    with pool.connection() as db:
        db.execute("UPDATE user_taste SET seen_ids = '{}' WHERE user_id = %s", (user_id,))


# ---------- products ----------

def count_products():
    with pool.connection() as db:
        return db.execute("SELECT count(*) FROM products").fetchone()[0]


def save_products(products, vectors):
    """Replace the whole catalogue with a new one.

    The delete and the inserts are one transaction: if anything fails,
    Postgres undoes all of it and the old catalogue stays.
    """
    # Postgres cannot store pandas' NaN in a text column, so blanks become None (NULL)
    products = products[PRODUCT_COLUMNS].astype(object)
    products = products.where(products.notna(), None)

    rows = []
    for i, product in enumerate(products.to_dict("records")):
        row = [product[column] for column in PRODUCT_COLUMNS]
        row.append(vectors[i].tolist())          # numpy numbers -> plain Python list
        rows.append(row)

    columns = ", ".join(PRODUCT_COLUMNS) + ", embedding"
    blanks = ", ".join(["%s"] * (len(PRODUCT_COLUMNS) + 1))

    with pool.connection() as db:                # commits at the end, undoes everything on error
        db.execute("DELETE FROM products")
        with db.cursor() as cur:
            cur.executemany(f"INSERT INTO products ({columns}) VALUES ({blanks})", rows)


def load_products():
    """Every active product as a dictionary: its details plus its own vector.

    Example: {"product_id": "SNITCH_859...", "title": "Slim Fit Shirt", ...,
              "vector": array of 512 numbers}
    The details and the numbers come from the same database row, so they always belong together.
    """
    with pool.connection() as db:
        cursor = db.cursor(row_factory=dict_row)          # each row comes back as a dictionary
        rows = cursor.execute(
            "SELECT " + ", ".join(PRODUCT_COLUMNS) + ", embedding FROM products WHERE is_active = TRUE"
        ).fetchall()

    products = []
    for row in rows:
        row["vector"] = np.array(row["embedding"], dtype="float32")   # the 512 numbers as numpy
        del row["embedding"]                                          # not needed twice
        products.append(row)
    return products