"""
Builds one feed of 20 products: 14 personalised, 2 trending, 2 sponsored, 2 wildcard.

The idea: compare the user's taste with every product they may see, then pick the best.
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
FEED_SIZE = 20

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


def match_score(product_vector, tastes):
    """How well a product matches the user: its match with the closest of their tastes."""
    best = -999
    for taste in tastes:
        match = float(np.dot(product_vector, taste))
        if match > best:
            best = match
    return best


def fits(product, counts):
    """True if adding this product keeps the feed within the variety limits."""
    if counts.get("category:" + str(product["category"]), 0) >= MAX_SAME_CATEGORY:
        return False
    if counts.get("type:" + str(product["subcategory"]), 0) >= MAX_SAME_TYPE:
        return False
    if counts.get("brand:" + str(product["brand"]), 0) >= MAX_SAME_BRAND:
        return False
    return True


def pick(candidates, how_many, why, feed, counts):
    """Add up to how_many products from candidates to the feed, in order, respecting the limits.
    feed is a dictionary: product_id -> why it was picked."""
    added = 0
    for product_id in candidates:
        if added == how_many:
            break
        if product_id in feed:
            continue
        product = catalog.products[product_id]
        if not fits(product, counts):
            continue
        feed[product_id] = why
        for key in ["category:" + str(product["category"]),
                    "type:" + str(product["subcategory"]),
                    "brand:" + str(product["brand"])]:
            counts[key] = counts.get(key, 0) + 1
        added = added + 1


def build_feed(tastes, gender, seen_ids):
    """Up to 20 products for this user, best match first. An empty list if nothing is left."""
    allowed = ALLOWED_GENDER[gender]
    seen = set(seen_ids)

    # 1. Compare the user's taste with every product they may see
    scores = {}                                    # product_id -> how well it matches
    for product_id, product in catalog.products.items():
        if product["gender"] not in allowed:
            continue
        if product_id in seen:
            continue
        scores[product_id] = match_score(product["vector"], tastes)

    if len(scores) == 0:
        return []

    # 2. Product ids from best match to worst
    ranked = sorted(scores, key=scores.get, reverse=True)

    feed = {}           # product_id -> why it was picked
    counts = {}         # how many of each category / type / brand so far

    # 3. Personalised: 14 at random from the best 100
    best = ranked[:PERSONALISED_POOL]
    random.shuffle(best)
    pick(best, FEED["personalised"], "personalised", feed, counts)

    # 4. Trending and sponsored: the best matching ones among those
    trending = []
    sponsored = []
    for product_id in ranked:
        if catalog.products[product_id]["is_trending"]:
            trending.append(product_id)
        if catalog.products[product_id]["sponsored"]:
            sponsored.append(product_id)
    pick(trending, FEED["trending"], "trending", feed, counts)
    pick(sponsored, FEED["sponsored"], "sponsored", feed, counts)

    # 5. Wildcard: random products, so the user can discover styles outside their taste
    leftover = []
    for product_id in ranked:
        if product_id not in feed:
            leftover.append(product_id)
    for product_id in random.sample(leftover, min(FEED["wildcard"], len(leftover))):
        feed[product_id] = "wildcard"

    # 6. If a slot could not be filled, top up with the best remaining matches.
    #    The variety limits are ignored here: a full feed is better than a gap.
    for product_id in ranked:
        if len(feed) >= FEED_SIZE:
            break
        if product_id not in feed:
            feed[product_id] = "topup"

    # 7. Turn the product ids into cards for the app, best match first
    chosen = sorted(feed, key=scores.get, reverse=True)
    cards = []
    for product_id in chosen:
        card = catalog.get_card(product_id)
        card["source"] = feed[product_id]
        card["score"] = round(scores[product_id], 3)
        cards.append(card)
    return cards