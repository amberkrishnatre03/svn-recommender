"""
Builds one feed of 20 products: 14 personalised, 2 trending, 2 sponsored, 2 wildcard.

Personalised and trending/sponsored cards are picked purely by best match score.
MMR (diversity) is only applied to wildcards, so they are genuinely different
from the rest of the feed rather than random noise.
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

# Each style's personalised cards are chosen from its best N matches
PERSONALISED_POOL = 100

# Trending and sponsored cards are chosen from the best N matches
DISCOVERY_POOL = 50

# Wildcard MMR weight: 0.0 = maximum diversity, 1.0 = best match (no diversity).
# Keep this low so wildcards show the user something genuinely new.
WILDCARD_MMR_WEIGHT = 0.1


def fits(product, counts):
    """True if adding this product keeps the feed within the variety limits."""
    if counts.get("category:" + str(product["category"]), 0) >= MAX_SAME_CATEGORY:
        return False
    if counts.get("type:" + str(product["subcategory"]), 0) >= MAX_SAME_TYPE:
        return False
    if counts.get("brand:" + str(product["brand"]), 0) >= MAX_SAME_BRAND:
        return False
    return True


def add(product_id, why, feed, counts):
    """Put a product in the feed and count it towards the variety limits."""
    product = catalog.products[product_id]
    feed[product_id] = why
    for key in ["category:" + str(product["category"]),
                "type:" + str(product["subcategory"]),
                "brand:" + str(product["brand"])]:
        counts[key] = counts.get(key, 0) + 1


def similarity_to_feed(product, feed):
    """How much this product looks like the most similar card already in the feed (0 if empty)."""
    most_similar = 0.0
    for chosen_id in feed:
        similarity = float(np.dot(product["vector"], catalog.products[chosen_id]["vector"]))
        if similarity > most_similar:
            most_similar = similarity
    return most_similar


def pick(candidates, match_of, how_many, why, feed, counts, pool=DISCOVERY_POOL, match_weight=1.0):
    """Add up to how_many products from candidates to the feed.

    match_weight=1.0 (default): pure best match, no diversity penalty.
    match_weight < 1.0: MMR mode — each pick balances match score against
    similarity to cards already in the feed, so picks are spread out visually.

    candidates must be sorted best match first. Only the first `pool` valid
    candidates are compared per pick so we never choose a poor match just to
    be different.
    match_of is a dict: product_id -> match score. feed: product_id -> why picked.
    """
    remaining = list(candidates)
    added = 0
    while added < how_many:
        best_id = None
        best_value = -999
        looked_at = 0
        for product_id in remaining:
            if product_id in feed:
                continue
            product = catalog.products[product_id]
            if not fits(product, counts):
                continue
            looked_at = looked_at + 1
            if looked_at > pool:
                break
            value = (match_weight * match_of[product_id]
                     - (1 - match_weight) * similarity_to_feed(product, feed))
            if value > best_value:
                best_value = value
                best_id = product_id

        if best_id is None:
            break

        add(best_id, why, feed, counts)
        remaining.remove(best_id)
        added = added + 1
    return added


def shares_by_points(points, total):
    """Split total cards between styles in proportion to their points, e.g. 5, 1, 1 -> 10, 2, 2."""
    exact = [total * p / sum(points) for p in points]
    shares = [int(x) for x in exact]
    by_leftover = sorted(range(len(points)), key=lambda i: exact[i] - shares[i], reverse=True)
    for i in by_leftover[:total - sum(shares)]:
        shares[i] = shares[i] + 1
    return shares


def build_feed(tastes, styles, points, gender, seen_ids):
    """Up to 20 products for this user. An empty list if nothing is left.
    styles[i] is the style of tastes[i], points[i] how many cards it earns."""
    allowed = ALLOWED_GENDER[gender]
    seen = set(seen_ids)

    styles_line_up = len(styles) == len(tastes)
    if len(points) != len(tastes):
        points = [1] * len(tastes)

    # 1. Score every unseen product against each taste vector
    scores = {}
    taste_scores = []
    for taste in tastes:
        taste_scores.append({})

    for product_id, product in catalog.products.items():
        if product["gender"] not in allowed:
            continue
        if product_id in seen:
            continue
        best = -999
        for number in range(len(tastes)):
            match = float(np.dot(product["vector"], tastes[number]))
            taste_scores[number][product_id] = match
            if match > best:
                best = match
        scores[product_id] = best

    if len(scores) == 0:
        return []

    # 2. Ranked lists: all products and products from main styles only
    ranked = sorted(scores, key=scores.get, reverse=True)
    main_styles = list(styles)
    if styles_line_up and len(styles) > 0:
        main_styles = []
        for number in range(len(styles)):
            if points[number] >= max(points) / 2:
                main_styles.append(styles[number])
    ranked_in_styles = [p for p in ranked if catalog.products[p]["style"] in main_styles]

    feed = {}
    counts = {}

    # 3. Personalised: pure best match per style, no MMR (match_weight=1.0 is the default)
    shares = shares_by_points(points, FEED["personalised"])
    for number in range(len(tastes)):
        share = shares[number]
        if share == 0:
            continue
        this_taste = taste_scores[number]
        candidates = [p for p in this_taste
                      if not styles_line_up or catalog.products[p]["style"] == styles[number]]
        best = sorted(candidates, key=this_taste.get, reverse=True)
        pick(best, this_taste, share, "personalised", feed, counts, PERSONALISED_POOL)

    # Fill any gap from main styles, still pure best match
    pick(ranked_in_styles, scores, FEED["personalised"] - len(feed),
         "personalised", feed, counts, PERSONALISED_POOL)

    # 4. Trending and sponsored: pure best match, no MMR
    for name, flag in [("trending", "is_trending"), ("sponsored", "sponsored")]:
        in_styles = [p for p in ranked_in_styles if catalog.products[p][flag]]
        others    = [p for p in ranked           if catalog.products[p][flag]]
        added = pick(in_styles, scores, FEED[name], name, feed, counts)
        pick(others, scores, FEED[name] - added, name, feed, counts)

    # 5. Wildcards: MMR against the whole feed so they look genuinely different
    #    from everything already picked. Low match_weight = high diversity.
    wildcard_pool = [p for p in ranked if p not in feed]
    pick(wildcard_pool, scores, FEED["wildcard"], "wildcard", feed, counts,
         pool=len(wildcard_pool), match_weight=WILDCARD_MMR_WEIGHT)

    # 6. Top up if any slot is still empty, ignoring variety limits
    for product_id in ranked_in_styles + ranked:
        if len(feed) >= FEED_SIZE:
            break
        if product_id not in feed:
            feed[product_id] = "topup"

    # 7. Order: personalised first (shuffled, strongest style goes first),
    #    then trending, sponsored, wildcard, topup
    personalised = [p for p in feed if feed[p] == "personalised"]
    random.shuffle(personalised)
    if styles_line_up and len(styles) > 0:
        strongest = styles[points.index(max(points))]
        for i in range(len(personalised)):
            if catalog.products[personalised[i]]["style"] == strongest:
                personalised.insert(0, personalised.pop(i))
                break
    chosen = personalised + [p for p in feed if feed[p] != "personalised"]

    cards = []
    for product_id in chosen:
        card = catalog.get_card(product_id)
        card["source"] = feed[product_id]
        card["score"] = round(scores[product_id], 3)
        cards.append(card)
    return cards