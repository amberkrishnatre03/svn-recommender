"""
Remembers each user's taste between requests.

One table, one row per user:
    user_id   who they are
    gender    Male or Female
    vector    their taste vectors, stored flattened end to end
    seen_ids  the last SEEN_LIMIT products they have been shown
"""
import os
import numpy as np
from psycopg_pool import ConnectionPool

ADDRESS = os.getenv("DATABASE_URL")     # stored inside render variable as DATABASE_URL

# Each taste vector is 512 numbers. A user has one per style they picked,
# so we store them flattened end to end and reshape them on the way out.
VECTOR_SIZE = 512

# How many recently seen products we remember per user.
#
# Without a limit this list grows forever, and we read the whole thing back
# on every single feed request. Worse, once a user has seen every product
# the feed returns nothing at all and they are stuck.
#
# Keeping only the most recent 500 means older products quietly become
# available again, so the feed never runs dry and repeats come back
# gradually instead of all at once after a reset.
SEEN_LIMIT = 500

# Reuse a small pool of database connections.
# Stale connections are checked before use and old/idle connections are recycled.
pool = ConnectionPool(
    ADDRESS,
    min_size=1,
    max_size=5,
    check=ConnectionPool.check_connection,
    max_idle=3000,
    max_lifetime=1800,
)


def connect():
    return pool.connection()


def create_table():
    """Run this once, when setting up."""
    with connect() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS user_taste (
                user_id  TEXT PRIMARY KEY,
                gender   TEXT NOT NULL,
                vector   FLOAT8[] NOT NULL,
                seen_ids TEXT[] NOT NULL DEFAULT '{}'
            )
        """)
    print("table ready")


def save_new_user(user_id, gender, tastes):
    """Called when a user finishes onboarding.

    If the user already exists, this wipes their taste and their history
    and starts them fresh. That is what "redo my onboarding" means.
    """
    numbers = [float(x) for x in tastes.flatten()]

    with connect() as db:
        db.execute("""
            INSERT INTO user_taste (user_id, gender, vector, seen_ids)
            VALUES (%s, %s, %s, '{}')
            ON CONFLICT (user_id) DO UPDATE
            SET gender = EXCLUDED.gender,
                vector = EXCLUDED.vector,
                seen_ids = '{}'
        """, (user_id, gender, numbers))


def get_user(user_id):
    """Returns gender, tastes, seen_ids. Returns None if the user is new."""
    with connect() as db:
        row = db.execute(
            "SELECT gender, vector, seen_ids FROM user_taste WHERE user_id = %s",
            (user_id,)
        ).fetchone()

    if row is None:
        return None

    gender, vector, seen_ids = row
    tastes = np.array(vector, dtype="float32").reshape(-1, VECTOR_SIZE)
    return gender, tastes, seen_ids


def update_user(user_id, tastes, newly_seen):
    """Save the new taste and add the products we just showed them.

    The || means "append" in Postgres. The slice at the end keeps only the
    most recent SEEN_LIMIT ids, so this list can never grow without bound.
    """
    numbers = [float(x) for x in tastes.flatten()]

    with connect() as db:
        db.execute("""
            UPDATE user_taste
            SET vector = %s,
                seen_ids = (seen_ids || %s)[
                    GREATEST(array_length(seen_ids || %s, 1) - %s + 1, 1)
                    :
                ]
            WHERE user_id = %s
        """, (numbers, newly_seen, newly_seen, SEEN_LIMIT, user_id))


def clear_seen(user_id):
    """Forget which products the user has been shown, but keep their taste.

    Used when someone has seen the whole catalogue and the feed runs dry.
    With SEEN_LIMIT in place this should rarely be needed, but it is kept
    for an explicit "start fresh" button.
    """
    with connect() as db:
        db.execute(
            "UPDATE user_taste SET seen_ids = '{}' WHERE user_id = %s",
            (user_id,)
        )

def get_trending(window, limit):
    """Product ids for one trending window, best first."""
    with connect() as db:
        rows = db.execute(
            'SELECT product_id FROM trending_products'
            ' WHERE "window" = %s ORDER BY "rank" LIMIT %s',
            (window, limit)
        ).fetchall()

    return [r[0] for r in rows]