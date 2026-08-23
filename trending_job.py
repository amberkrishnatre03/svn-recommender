"""
Works out which products are trending and saves them to the database.
Run every few hours:  python trending_job.py
"""

import os
import pandas as pd
import psycopg

WEIGHTS = {"Likes": 1, "AddToCart": 3, "Purchase": 5}


def main():
    now = pd.Timestamp.now(tz="UTC")
    events = pd.read_csv("data/mock_product_events.csv")
    events["createdAt"] = pd.to_datetime(events["createdAt"], utc=True)

    # score = how much the action counts, faded by how old it is
    events["weight"] = events["action"].map(WEIGHTS).fillna(0)
    days_old = (now - events["createdAt"]).dt.total_seconds() / 86400
    events["score"] = events["weight"] * 0.5 ** (days_old.clip(lower=0) / 3)

    rows = []
    for window, days in [("1d", 1), ("7d", 7), ("30d", 30)]:
        recent = events[events["createdAt"] >= now - pd.Timedelta(days=days)]

        ranked = (recent.groupby("product_id")["score"].sum()
                        .sort_values(ascending=False).head(200))

        for rank, (product_id, score) in enumerate(ranked.items(), start=1):
            rows.append((window, product_id, rank, float(score), now))

    with psycopg.connect(os.environ["DATABASE_URL"]) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS trending_products (
                "window" TEXT, product_id TEXT, "rank" INT,
                score FLOAT8, computed_at TIMESTAMPTZ)
        """)
        db.execute("DELETE FROM trending_products")
        db.cursor().executemany(
            'INSERT INTO trending_products ("window", product_id, "rank", score, computed_at)'
            ' VALUES (%s,%s,%s,%s,%s)', rows)
    print("saved", len(rows), "rows")


if __name__ == "__main__":
    main()