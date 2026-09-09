"""
Builds the feed: 12 personalised, 4 trending, 1 sponsored, 3 wildcard.
"""

import random

import numpy as np

from app import catalog

# How many of each kind of card in one feed.
FEED = {
    "personalised": 12,
    "trending": 4,
    "sponsored": 1,
    "wildcard": 3,
}

# Limits so one feed is not all t-shirts from one brand.
# A cap must be big enough that the feed can actually be filled. With a feed
# of 20 and only 4 brands, a cap of 4 could never be met, the topup pass would
# take over and every limit here would be ignored.
MAX_SAME_CATEGORY = 7     # at most 7 Topwear and 7 bottomwear in a feed of 20
MAX_SAME_TYPE = 5          # at most 5 T-Shirts, 5 Shirts, and so on
MAX_SAME_BRAND = 8         # at most 8 from any one brand

# How many of the best matching products to sample the personalised cards
# from. The 20th best match is usually as good as the 3rd, so always taking
# them strictly in order makes the feed look the same every time for no real
# gain in relevance.
PERSONALISED_POOL = 200

# How much a brand the user picked at onboarding gets pushed up the ranking.
# This is a nudge, not a filter: other brands still appear, they just have to
# score better to get in. A hard filter would shrink the pool so much that the
# feed would start repeating within days.
BRAND_BOOST = 0.15

# Male users see Men's products, female users see Women's.
# Add "Unisex" to a list later when we bring unisex products back.
ALLOWED_GENDER = {
    "Male": ["Men"],
    "Female": ["Women"],
}


def find_allowed_rows(gender, seen_ids):
    """Every product this user is allowed to see right now.

    This is the hard filter. Gender is a business rule and similarity
    can never override it.
    """
    allowed = ALLOWED_GENDER[gender]

    keep = catalog.products["gender"].isin(allowed)
    keep = keep & catalog.products["is_active"]
    keep = keep & ~catalog.products["product_id"].isin(seen_ids)

    return np.where(keep)[0]


def pick_with_variety(rows, scores, how_many, chosen, counts, pool_size=None):
    """Take the best products, but keep the feed varied.

    counts holds how many of each category, subcategory and brand are
    already in this feed, so the limits apply across the whole feed and
    not just inside one section.

    pool_size shuffles the best N before picking, so the same user does
    not see the same products in the same order every time.
    """
    picked = []
    already = set(chosen)

    # Best score first.
    order = np.argsort(-scores)

    if pool_size:
        order = order[:pool_size]
        np.random.shuffle(order)

    for row in rows[order]:
        if len(picked) == how_many:
            break

        if row in already:
            continue

        category = catalog.category_of[row]
        product_type = catalog.subcategory_of[row]
        brand = catalog.brand_of[row]

        if counts.get(category, 0) >= MAX_SAME_CATEGORY:
            continue

        if counts.get(product_type, 0) >= MAX_SAME_TYPE:
            continue

        if counts.get(brand, 0) >= MAX_SAME_BRAND:
            continue

        picked.append(row)
        counts[category] = counts.get(category, 0) + 1
        counts[product_type] = counts.get(product_type, 0) + 1
        counts[brand] = counts.get(brand, 0) + 1

    return picked


def build_feed(user_tastes, gender, seen_ids, preferred_brands=None):
    """Return one feed of products for this user."""
    allowed_rows = find_allowed_rows(gender, seen_ids)

    # Nothing left to show at all.
    if len(allowed_rows) == 0:
        return []

    # Score each product by its best match to any of the user's tastes.
    scores = (catalog.vectors[allowed_rows] @ user_tastes.T).max(axis=1)

    # Brands they picked at onboarding get a small bump, so their favourites
    # surface more often without pushing everything else out of the feed.
    if preferred_brands:
        brands_here = catalog.products["brand"].values[allowed_rows]
        scores = scores + np.isin(brands_here, preferred_brands) * BRAND_BOOST

    # So we can look up any row's score later.
    score_of = dict(zip(allowed_rows, scores))

    chosen = []
    sources = []
    counts = {}

    def add(rows, name):
        chosen.extend(rows)
        sources.extend([name] * len(rows))

    # 1. Personalised: sampled from the products closest to their taste.
    add(pick_with_variety(allowed_rows, scores, FEED["personalised"],
                          chosen, counts, pool_size=PERSONALISED_POOL),
        "personalised")

    # 2. Trending: popular right now, still matched to their taste.
    is_trending = catalog.products["is_trending"].values[allowed_rows]
    add(pick_with_variety(allowed_rows[is_trending], scores[is_trending],
                          FEED["trending"], chosen, counts), "trending")

    # 3. Sponsored: paid placement, still the best matching one.
    is_sponsored = catalog.products["sponsored"].values[allowed_rows]
    add(pick_with_variety(allowed_rows[is_sponsored], scores[is_sponsored],
                          FEED["sponsored"], chosen, counts), "sponsored")

    # 4. Wildcard: something random, so the user can discover new styles.
    # Score is ignored on purpose. If we only ever show products near their
    # current taste we can sharpen that taste but never find out it moved.
    taken = set(chosen)
    leftover = [row for row in allowed_rows if row not in taken]
    if leftover:
        how_many = min(FEED["wildcard"], len(leftover))
        picks = np.random.choice(leftover, size=how_many, replace=False)
        add([int(row) for row in picks], "wildcard")

    # If a slot could not be filled, top up with anything left.
    # Variety is dropped here on purpose: a full feed beats a varied gap.
    wanted = sum(FEED.values())
    taken = set(chosen)
    for row in allowed_rows[np.argsort(-scores)]:
        if len(chosen) >= wanted:
            break
        if row not in taken:
            add([row], "topup")
            taken.add(row)

    # Turn the chosen rows into products, telling the app why each was picked.
    feed = []
    for row, source in zip(chosen, sources):
        product = catalog.get_product(row)
        product["source"] = source
        product["score"] = round(float(score_of[row]), 3)
        feed.append(product)

    # Best match first, weakest last. Without this the stack would end on trending, sponsored and wildcard, because those are added after the personalised cards.
    feed.sort(key=lambda p: -p["score"])             # sorted by scores lambda just reduces the function
    return feed