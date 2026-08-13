"""
Builds the feed: 8 personalised, 1 trending, 1 sponsored, 1 wildcard.
"""

import random

import numpy as np

from app import catalog

# How many of each kind of card in one feed.
FEED = {
    "personalised": 8,
    "trending": 1,
    "sponsored": 1,
    "wildcard": 1,
}

# Limits so one feed is not all t-shirts.
MAX_SAME_CATEGORY = 8      # at most 5 Topwear in a feed of 11
MAX_SAME_TYPE = 3          # at most 3 T-Shirts, 3 Hoodies, and so on

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


def pick_with_variety(rows, scores, how_many, chosen, counts):
    """Take the best products, but keep the feed varied.

    counts holds how many of each category and subcategory are already
    in this feed, so the limits apply across the whole feed and not
    just inside one section.
    """
    picked = []
    already = set(chosen)

    # Best score first.
    for row in rows[np.argsort(-scores)]:
        if len(picked) == how_many:
            break

        if row in already:
            continue

        category = catalog.category_of[row]
        product_type = catalog.subcategory_of[row]

        if counts.get(category, 0) >= MAX_SAME_CATEGORY:
            continue

        if counts.get(product_type, 0) >= MAX_SAME_TYPE:
            continue

        picked.append(row)
        counts[category] = counts.get(category, 0) + 1
        counts[product_type] = counts.get(product_type, 0) + 1

    return picked

def build_feed(user_tastes, gender, seen_ids):
    """Return one feed of products for this user."""
    allowed_rows = find_allowed_rows(gender, seen_ids)

    # Nothing left to show at all.
    if len(allowed_rows) == 0:
        return []

    # Score every allowed product against this user's taste.
        # Score each product by its best match to any of the user's tastes.
    scores = (catalog.vectors[allowed_rows] @ user_tastes.T).max(axis=1)

    # So we can look up any row's score later.
    score_of = dict(zip(allowed_rows, scores))

    chosen = []
    sources = []
    counts = {}

    def add(rows, name):
        chosen.extend(rows)
        sources.extend([name] * len(rows))

    # 1. Personalised: the products closest to their taste.
    add(pick_with_variety(allowed_rows, scores, FEED["personalised"], chosen, counts),
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
    taken = set(chosen)
    leftover = [row for row in allowed_rows if row not in taken]
    if leftover:
        add([int(np.random.choice(leftover))], "wildcard")

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

    # The special cards are added last, so without this every stack would end
    # on trending, sponsored and wildcard. Spread them through instead, but
    # keep the strongest personalised card first so the stack opens well.
    if len(feed) > 2:
        rest = feed[1:]
        random.shuffle(rest)
        feed = [feed[0]] + rest

    return feed