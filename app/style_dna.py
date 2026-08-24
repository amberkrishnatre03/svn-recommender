"""
Works out a user's style breakdown from their recent likes.

This is only for the slider on the profile page. Nothing else reads it.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
from app import catalog, database


EVENTS_FILE = Path(__file__).parent.parent / "data" / "mock_product_events.csv"

# A purchase says more about someone's taste than a swipe does.
# Dislikes are left out: this is "what you are into", not what you are not.
POINTS = {"Likes": 1, "AddToCart": 3, "Purchase": 5}

HOW_MANY = 50
import os

# Where the real interaction log comes from. Falls back to the mock file
# so this works before backend's API exists.
EVENTS_URL = os.getenv("EVENTS_URL")

def recent_interactions(user_id):
    """This user's most recent positive interactions.

    When EVENTS_URL is set we ask backend for just this user's rows,
    which is why the limit cannot hide someone's history. Without it
    we fall back to the mock file for testing.
    """
    if EVENTS_URL:
        events = pd.read_csv(f"{EVENTS_URL}?user_id={user_id}&limit={HOW_MANY}")
    else:
        events = pd.read_csv(EVENTS_FILE)
        events = events[events["user_id"] == user_id]

    mine = events[events["action"].isin(POINTS)]
    return mine.tail(HOW_MANY)

def build(user_id):
    """Their taste as percentages, biggest first."""
    mine = recent_interactions(user_id)

    points = Counter()
    for _, row in mine.iterrows():
        if not catalog.has_product(row["product_id"]):
            continue
        row_number = catalog.row_of_product[row["product_id"]]
        style = catalog.products["style"].iloc[row_number]
        points[style] += POINTS[row["action"]]

    total = sum(points.values())

    # Nothing to count yet. Fall back to the styles they picked at
    # onboarding, split evenly, so a new user still sees something.
    if total == 0:
        chosen = database.get_styles(user_id)
        if not chosen:
            return {"based_on": 0, "dna": []}

        share = round(100 / len(chosen))
        return {
            "based_on": 0,
            "from": "onboarding",
            "dna": [{"style": s, "percent": share} for s in chosen],
        }

    dna = [
        {"style": style, "percent": round(100 * score / total)}
        for style, score in points.most_common(3)
    ]

    return {"based_on": len(mine), "dna": dna}