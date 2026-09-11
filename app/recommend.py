"""
Builds one feed of 20 products: 14 personalised, 2 trending, 2 sponsored, 2 wildcard.
"""

import random

import numpy as np

from app import catalog

# How many of each kind of card in one feed
FEED = {
    "personalised": 14,
    "trending": 2,
    "sponsored": 2,
    "wildcard": 2,
}
FEED_SIZE = sum(FEED.values())

# Which product genders each user gender may see
ALLOWED_GENDER = {
    "Male": ["Men"],
    "Female": ["Women"],
    "Other": ["Men", "Women"],
}

# Limits so one feed is not all T-shirts from one brand
MAX_SAME_CATEGORY = 12
MAX_SAME_TYPE = 8
MAX_SAME_BRAND = 8

# Personalised cards are drawn at random from the best 100 matches, not always
# the top 14 in order, so the feed does not look the same every time.
PERSONALISED_POOL = 100


def allowed_rows(gender, seen_ids):
    """Row numbers of products this user may see: right gender, not seen yet."""
    keep = np.isin(catalog.gender_of, ALLOWED_GENDER[gender])
    for product_id in seen_ids:
        row = catalog.row_of.get(product_id)
        if row is not None:
            keep[row] = False
    return np.where(keep)[0]


def fits(row, counts):
    """True if adding this product keeps the feed within the variety limits."""
    return (counts.get(catalog.category_of[row], 0) < MAX_SAME_CATEGORY
            and counts.get(catalog.type_of[row], 0) < MAX_SAME_TYPE
            and counts.get(catalog.brand_of[row], 0) < MAX_SAME_BRAND)


def pick(rows, scores, how_many, chosen, counts, pool_size=None):
    """Pick up to how_many rows, best score first, skipping chosen ones and respecting the limits."""
    order = rows[np.argsort(-scores)]               # best match first
    if pool_size:
        order = list(order[:pool_size])
        random.shuffle(order)                       # mix up the best ones

    picked = []
    for row in order:
        if len(picked) == how_many:
            break
        if row in chosen or not fits(row, counts):
            continue
        picked.append(row)
        chosen.add(row)
        for key in (catalog.category_of[row], catalog.type_of[row], catalog.brand_of[row]):
            counts[key] = counts.get(key, 0) + 1
    return picked


def build_feed(tastes, gender, seen_ids):
    """Up to 20 products for this user, best match first. Empty list if nothing is left."""
    rows = allowed_rows(gender, seen_ids)
    if len(rows) == 0:
        return []

    # Each product's score is how well it matches the closest of the user's tastes
    scores = (catalog.vectors[rows] @ tastes.T).max(axis=1)
    score_of = dict(zip(rows, scores))

    chosen = set()      # rows already in the feed
    counts = {}         # how many of each category / type / brand so far
    feed = []           # (row, why it was picked)

    for row in pick(rows, scores, FEED["personalised"], chosen, counts, PERSONALISED_POOL):
        feed.append((row, "personalised"))

    trending = catalog.is_trending[rows]
    for row in pick(rows[trending], scores[trending], FEED["trending"], chosen, counts):
        feed.append((row, "trending"))

    sponsored = catalog.is_sponsored[rows]
    for row in pick(rows[sponsored], scores[sponsored], FEED["sponsored"], chosen, counts):
        feed.append((row, "sponsored"))

    # Wildcard: random products, so the user can discover styles outside their taste
    leftover = [row for row in rows if row not in chosen]
    for row in random.sample(leftover, min(FEED["wildcard"], len(leftover))):
        feed.append((row, "wildcard"))
        chosen.add(row)

    # If any slot could not be filled, top up with the best remaining matches.
    # The variety limits are ignored here: a full feed is better than a gap.
    for row in rows[np.argsort(-scores)]:
        if len(feed) >= FEED_SIZE:
            break
        if row not in chosen:
            feed.append((row, "topup"))
            chosen.add(row)

    cards = []
    for row, source in feed:
        card = catalog.get_card(row)
        card["source"] = source
        card["score"] = round(float(score_of[row]), 3)
        cards.append(card)

    cards.sort(key=lambda card: -card["score"])    # best match first
    return cards