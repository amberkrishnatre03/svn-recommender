"""
Works out which products are trending and saves them to the database.
Run every few hours:  python trending_job.py
"""

import os
import pandas as pd
import psycopg

from app import catalog

WEIGHTS = {"Likes": 1, "AddToCart": 3, "Purchase": 5}


def main():
    now = pd.Timestamp.now(tz="UTC")
    events = pd.read_csv("data/mock_product_events.csv")
    events["createdAt"] = pd.to_datetime(events["createdAt"], utc=True)

    # score = how much the action counts, faded by how old it is
    events["weight"] = events["action"].map(WEIGHTS).fillna(0)
    days_old = (now - events["createdAt"]).dt.total_seconds() / 86400
    events["score"] = events["weight"] * 0.5 ** (days_old.clip(lower=0) / 3)

    # brand, style and category are copied in so the app can filter on them
    # without us having to look anything up at request time
    info = catalog.products.set_index("product_id")

    rows = []
    for window, days in [("1d", 1), ("7d", 7), ("30d", 30)]:
        recent = events[events["createdAt"] >= now - pd.Timedelta(days=days)]

        ranked = (recent.groupby("product_id")["score"].sum()
                        .sort_values(ascending=False).head(200))

        for rank, (product_id, score) in enumerate(ranked.items(), start=1):
            if product_id not in info.index:
                continue
            p = info.loc[product_id]
            rows.append((window, product_id, rank, float(score),
                         p["brand"], p["style"], p["category"], now))

    with psycopg.connect(os.environ["DATABASE_URL"]) as db:
        db.execute("DROP TABLE IF EXISTS trending_products")
        db.execute("""
            CREATE TABLE trending_products (
                "window" TEXT, product_id TEXT, "rank" INT, score FLOAT8,
                brand TEXT, style TEXT, category TEXT, computed_at TIMESTAMPTZ)
        """)
        db.cursor().executemany(
            'INSERT INTO trending_products'
            ' ("window", product_id, "rank", score, brand, style, category, computed_at)'
            ' VALUES (%s,%s,%s,%s,%s,%s,%s,%s)', rows)

    print("saved", len(rows), "rows")


if __name__ == "__main__":
    main()