"""
Builds one feed of 20 products: 14 personalised, 2 trending, 2 sponsored, 2 wildcard.

Score every unseen product against the user's taste, pick the best matches.
MMR (diversity) is only used for wildcards so they show something new.
"""

import random

import numpy as np

from app import catalog

# --- tuning knobs, all in one place ---

FEED_SIZE   = 20
PERSONALISED = 14
TRENDING     = 2
SPONSORED    = 2
WILDCARD     = 2

# How many top candidates to look at when picking personalised cards.
# Higher = slower but more accurate. 200 is plenty for 5k products.
CANDIDATE_POOL = 200

# Wildcard MMR weight. 0.0 = wildcard must look totally different from the feed.
# 1.0 = no diversity, just best match. Keep low so wildcards feel like discovery.
WILDCARD_MMR = 0.1

# Variety limits per feed so it's not all one brand or one type
MAX_SAME_CATEGORY = 12
MAX_SAME_TYPE     = 8
MAX_SAME_BRAND    = 8

# Which product genders each user gender may see
ALLOWED_GENDER = {
    "Male":   ["Men"],
    "Female": ["Women"],
    "Other":  ["Men", "Women"],
}


# --- helpers ---

def under_limit(product, counts):
    """True if this product keeps the feed within variety limits."""
    if counts.get("category:" + str(product["category"]), 0) >= MAX_SAME_CATEGORY:
        return False
    if counts.get("type:"     + str(product["subcategory"]), 0) >= MAX_SAME_TYPE:
        return False
    if counts.get("brand:"    + str(product["brand"]), 0) >= MAX_SAME_BRAND:
        return False
    return True


def add(product_id, why, feed, counts):
    """Add a product to the feed and update variety counts."""
    product = catalog.products[product_id]
    feed[product_id] = why
    counts["category:" + str(product["category"])] = counts.get("category:" + str(product["category"]), 0) + 1
    counts["type:"     + str(product["subcategory"])] = counts.get("type:" + str(product["subcategory"]), 0) + 1
    counts["brand:"    + str(product["brand"])] = counts.get("brand:" + str(product["brand"]), 0) + 1


def most_similar_in_feed(product, feed):
    """Cosine similarity between this product and the most similar card already in the feed."""
    best = 0.0
    for chosen_id in feed:
        sim = float(np.dot(product["vector"], catalog.products[chosen_id]["vector"]))
        if sim > best:
            best = sim
    return best


def pick(candidates, scores, how_many, why, feed, counts, pool=CANDIDATE_POOL, mmr=1.0):
    """Pick up to how_many products from candidates and add them to the feed.

    mmr=1.0  → pure best match (no diversity penalty), used for personalised/trending/sponsored.
    mmr<1.0  → MMR mode: each pick balances match score vs similarity to what's already in feed.
               Used for wildcards so they look genuinely different from the rest of the feed.

    Only the first `pool` valid candidates are considered per pick so slow products
    at the bottom of the list never sneak in just for being different.
    """
    remaining = list(candidates)
    added = 0
    while added < how_many:
        best_id    = None
        best_value = -999
        looked_at  = 0

        for product_id in remaining:
            if product_id in feed:
                continue
            product = catalog.products[product_id]
            if not under_limit(product, counts):
                continue
            looked_at += 1
            if looked_at > pool:
                break

            value = mmr * scores[product_id] - (1 - mmr) * most_similar_in_feed(product, feed)
            if value > best_value:
                best_value = value
                best_id    = product_id

        if best_id is None:
            break

        add(best_id, why, feed, counts)
        remaining.remove(best_id)
        added += 1
    return added


# --- main feed builder ---

def build_feed(tastes, styles, style_scores, gender, seen_ids):
    """Return up to 20 product cards for this user. Returns [] if nothing is left to show."""

    allowed = ALLOWED_GENDER.get(gender, ["Men", "Women"])
    seen    = set(seen_ids)

    # 1. Score every unseen product the user is allowed to see.
    #    Score = best dot product across all of the user's taste vectors.
    #    (dot product of two unit vectors = cosine similarity = visual similarity)
    scores = {}
    for product_id, product in catalog.products.items():
        if product["gender"] not in allowed:
            continue
        if product_id in seen:
            continue
        best = max(float(np.dot(product["vector"], taste)) for taste in tastes)
        scores[product_id] = best

    if not scores:
        return []

    # 2. Sort all candidates best match first — this one list drives everything below
    ranked = sorted(scores, key=scores.get, reverse=True)

    feed   = {}   # product_id -> why it was picked
    counts = {}   # variety limit counters

    # 3. Personalised: best 14 matches, pure score, no diversity penalty
    pick(ranked, scores, PERSONALISED, "personalised", feed, counts)

    # 4. Trending: best matching products that have is_trending = True
    trending = [p for p in ranked if catalog.products[p]["is_trending"]]
    pick(trending, scores, TRENDING, "trending", feed, counts)

    # 5. Sponsored: best matching products that have sponsored = True
    sponsored = [p for p in ranked if catalog.products[p]["sponsored"]]
    pick(sponsored, scores, SPONSORED, "sponsored", feed, counts)

    # 6. Wildcards: MMR so each wildcard looks different from the whole feed.
    #    This is the only place MMR is used — wildcards are meant to surprise.
    pick(ranked, scores, WILDCARD, "wildcard", feed, counts, mmr=WILDCARD_MMR)

    # 7. Top up to FEED_SIZE if any slot is still empty (variety limits hit, etc.)
    #    Ignores variety limits so the feed is never short.
    for product_id in ranked:
        if len(feed) >= FEED_SIZE:
            break
        if product_id not in feed:
            feed[product_id] = "topup"

    # 8. Order: personalised first (shuffled so styles mix), then the rest
    personalised = [p for p in feed if feed[p] == "personalised"]
    random.shuffle(personalised)
    rest = [p for p in feed if feed[p] != "personalised"]

    cards = []
    for product_id in personalised + rest:
        card = catalog.get_card(product_id)
        card["source"] = feed[product_id]
        card["score"]  = round(scores[product_id], 3)
        cards.append(card)
    return cards