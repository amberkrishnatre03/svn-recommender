"""
Builds one feed of 20 products: 14 personalised, 2 trending, 2 sponsored, 2 wildcard.

The idea: compare the user's taste with every product they may see, then pick products
that match well AND are different from each other (MMR), so the stack isn't six copies
of the same black shirt.
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

# Each style's cards are chosen from its best 100 matches
PERSONALISED_POOL = 100

# MMR: how to weigh "matches the user" against "different from the cards already picked".
# 1.0 = only match (similar-looking cards), 0.0 = only different (random-looking cards).
MATCH_WEIGHT = 0.5

# Trending and sponsored cards are chosen from the best 50 matches
DISCOVERY_POOL = 50



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


def pick(candidates, match_of, how_many, why, feed, counts, pool=DISCOVERY_POOL):
    """Add up to how_many products from candidates to the feed, one at a time (MMR).

    Each time, take the candidate with the best value:
        MATCH_WEIGHT * how well it matches the user
        - (1 - MATCH_WEIGHT) * how much it looks like a card already in the feed
    so every new card is a good match that is also different from what's already there.
    candidates must be sorted best match first. Only the first `pool` candidates that still
    fit the variety limits are compared, so we never pick a poor match just for being different.
    match_of is a dictionary: product_id -> how well it matches. feed: product_id -> why picked.
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
            value = (MATCH_WEIGHT * match_of[product_id]
                     - (1 - MATCH_WEIGHT) * similarity_to_feed(product, feed))
            if value > best_value:
                best_value = value
                best_id = product_id

        if best_id is None:                        # nothing left that fits
            break

        add(best_id, why, feed, counts)
        remaining.remove(best_id)
        added = added + 1
    return added


def shares_by_points(points, total):
    """Split total cards between styles in proportion to their points, e.g. 5, 1, 1 -> 10, 2, 2."""
    exact = [total * p / sum(points) for p in points]
    shares = [int(x) for x in exact]
    # hand the cards left over to the styles that were closest to the next whole card
    by_leftover = sorted(range(len(points)), key=lambda i: exact[i] - shares[i], reverse=True)
    for i in by_leftover[:total - sum(shares)]:
        shares[i] = shares[i] + 1
    return shares


def build_feed(tastes, styles, points, gender, seen_ids):
    """Up to 20 products for this user. An empty list if nothing is left.
    styles[i] is the style of tastes[i], points[i] how many cards it earns."""
    allowed = ALLOWED_GENDER[gender]
    seen = set(seen_ids)

    # Users saved by older code can have styles that don't line up with their tastes.
    # For them, each taste gets an equal share drawn from every product, as before.
    styles_line_up = len(styles) == len(tastes)
    if len(points) != len(tastes):
        points = [1] * len(tastes)

    # 1. Compare every product the user may see with each of their tastes (one per style)
    scores = {}                                    # product_id -> its best match with any taste
    taste_scores = []                              # one dictionary per taste: product_id -> match
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

    # 2. Product ids from best match to worst, and the same list for the user's main styles:
    #    styles with at least half the points of their strongest style. At the start that's
    #    only the styles they picked; a neighbour style they keep liking becomes main later.
    #    Trending, sponsored and top-up cards come from main styles first.
    ranked = sorted(scores, key=scores.get, reverse=True)
    main_styles = list(styles)                     # old users whose data doesn't line up: all styles
    if styles_line_up and len(styles) > 0:
        main_styles = []
        for number in range(len(styles)):
            if points[number] >= max(points) / 2:
                main_styles.append(styles[number])
    ranked_in_styles = []
    for product_id in ranked:
        if catalog.products[product_id]["style"] in main_styles:
            ranked_in_styles.append(product_id)

    feed = {}           # product_id -> why it was picked
    counts = {}         # how many of each category / type / brand so far

    # 3. Personalised: each style gets a share of the 14 cards by its points. A style's
    #    share only uses products of that style, chosen from its best 100 matches with MMR.
    #    Without the shares, a style whose products all look alike (e.g. slim-fit shirts)
    #    scores higher and takes over the feed, even for users who never picked it.
    shares = shares_by_points(points, FEED["personalised"])
    for number in range(len(tastes)):
        share = shares[number]
        if share == 0:
            continue
        this_taste = taste_scores[number]
        candidates = []
        for product_id in this_taste:
            if not styles_line_up or catalog.products[product_id]["style"] == styles[number]:
                candidates.append(product_id)
        best = sorted(candidates, key=this_taste.get, reverse=True)
        pick(best, this_taste, share, "personalised", feed, counts, PERSONALISED_POOL)

    # A share can come up short (variety limits, or a style running out of products).
    # Fill the gap from the best matches in the chosen styles.
    pick(ranked_in_styles, scores, FEED["personalised"] - len(feed),
         "personalised", feed, counts, PERSONALISED_POOL)

    # 4. Trending and sponsored: the best matching ones, from the chosen styles first.
    #    Other styles are only used if the chosen styles have none left.
    for name, flag in [("trending", "is_trending"), ("sponsored", "sponsored")]:
        in_styles = [product_id for product_id in ranked_in_styles if catalog.products[product_id][flag]]
        others = [product_id for product_id in ranked if catalog.products[product_id][flag]]
        added = pick(in_styles, scores, FEED[name], name, feed, counts)
        pick(others, scores, FEED[name] - added, name, feed, counts)

    # 5. Wildcard: random products, so the user can discover styles outside their taste.
    #    They follow the variety limits too, so they can't push topwear past 12.
    for _ in range(FEED["wildcard"]):
        leftover = []
        for product_id in ranked:
            if product_id not in feed and fits(catalog.products[product_id], counts):
                leftover.append(product_id)
        if len(leftover) > 0:
            add(random.choice(leftover), "wildcard", feed, counts)

    # 6. If a slot could not be filled, top up with the best remaining matches, from the
    #    chosen styles first. The variety limits are ignored here: a full feed is better than a gap.
    for product_id in ranked_in_styles + ranked:
        if len(feed) >= FEED_SIZE:
            break
        if product_id not in feed:
            feed[product_id] = "topup"

    # 7. Order: personalised first, in mixed order so one style doesn't come in a row,
    #    then trending, sponsored, wildcard and any top-up cards at the end.
    #    The first card is from the user's strongest style.
    personalised = [product_id for product_id in feed if feed[product_id] == "personalised"]
    random.shuffle(personalised)
    if styles_line_up and len(styles) > 0:
        strongest = styles[points.index(max(points))]
        for i in range(len(personalised)):
            if catalog.products[personalised[i]]["style"] == strongest:
                personalised.insert(0, personalised.pop(i))
                break
    chosen = personalised + [product_id for product_id in feed if feed[product_id] != "personalised"]

    cards = []
    for product_id in chosen:
        card = catalog.get_card(product_id)
        card["source"] = feed[product_id]
        card["score"] = round(scores[product_id], 3)
        cards.append(card)
    return cards