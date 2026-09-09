"""
Works out which products are trending and saves them to the database.
Run every few hours:  python trending_job.py
"""

import os
import pandas as pd
import psycopg

from app import catalog

# Where the real interaction log comes from. The mock file below is only
# a stand in so this can be tested before backend's API exists.
EVENTS_URL = os.getenv("EVENTS_URL")           # getenv reads the env variable

WEIGHTS = {"Likes": 1, "AddToCart": 3, "Purchase": 5}       # weights dictionary


def main():
    now = pd.Timestamp.now(tz="UTC")                                         # timestamp captured every run
    events = pd.read_csv(EVENTS_URL or "data/mock_product_events.csv")        # reads the events table csv either from url or mockcsv either
    events["createdAt"] = pd.to_datetime(events["createdAt"], utc=True)          # convert created at column to utc for easiest subtraction

    # score = how much the action counts, faded by how old it is
    events["weight"] = events["action"].map(WEIGHTS).fillna(0)                   #new weights column mapping to weights , so now action is converted to weights
    days_old = (now - events["createdAt"]).dt.total_seconds() / 86400            # Event happened 0 days ago → days_old = 0 / this is variable
    events["score"] = events["weight"] * 0.5 ** (days_old.clip(lower=0) / 3)      # new score column / gives the recency score per interaction
                                                                                  # recency score basically
    # most_liked ignores the fading and just counts likes
    events["likes"] = (events["action"] == "Likes").astype(int)                # new column of likes that stores all the liked products

    # brand, style and category are copied in so the app can filter on them
    # without us having to look anything up at request time                   # basically craetes dataframe index
    info = catalog.products.set_index("product_id")                           # catalogue lookup need to do this because further functions require
                                                                              # info.loc[some_id],, after this and get all information of that product using this function
    rows = []                                                                 # empty list
    for window, days in [("1d", 1), ("7d", 7), ("30d", 30), ("most_liked", 30)]:        # runs the loop 4 times ( #1 - for window = 1day and day = 1,, #2 - for window = 7d and days = 7 etcc.. lastly for window = most liked and days = 30)
        recent = events[events["createdAt"] >= now - pd.Timedelta(days=days)]            # keeps interactions of only last 7 days or 1 day or 30d
                                                                                      # eg starts from 1 day interactions then keeps all interactions of days 1
        # trending fades old activity away, most_liked does not, so one
        # means "hot right now" and the other means "loved overall"
        column = "likes" if window == "most_liked" else "score"               # if else in one line this means
                                                                              # colum will = likes if window is most liked else its gonna be score
        ranked = (recent.groupby("product_id")[column].sum()                   # sums up the scores for everywhere where they are scored and
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