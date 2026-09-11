"""
A user's taste, as vectors.

A user has one taste vector per style, plus points per style. Keeping tastes
separate means someone who likes both Formal and Streetwear gets both, instead
of one average that sits in between and matches neither. Points decide how many
cards each style gets: styles they like grow, styles they skip fade.
"""

import numpy as np

from app import catalog, database

# How far one action moves the user's taste towards (or away from) a product.
# Unknown action names are ignored, so the spelling must match what the app sends.
WEIGHTS = {
    "dislike":      -0.05,
    "view":          0.03,
    "like":          0.15,
    "add_to_cart":   0.28,
    "purchase":      0.35,
}

# Styles that go together. Picking a style also gives a few cards from its neighbours,
# so a user who picks only Oversized still sees some Streetwear and Casual.
# This is a product decision, not maths: edit it freely.
NEIGHBOURS = {
    "Oversized":  ["Streetwear", "Casual"],
    "Streetwear": ["Oversized", "Athleisure"],
    "Athleisure": ["Streetwear", "Casual"],
    "Casual":     ["Oversized", "Athleisure", "Old Money"],
    "Old Money":  ["Formal", "Casual"],
    "Formal":     ["Old Money", "Partywear"],
    "Partywear":  ["Formal", "Streetwear"],
}

# Points per style decide its share of the 14 personalised cards.
# Oversized picked (5) + Streetwear and Casual as neighbours (0.5 each) = about 12 + 1 + 1 cards.
PICKED_POINTS = 5
NEIGHBOUR_POINTS = 0.5
POINTS = {"dislike": -0.5, "view": 0, "like": 1, "add_to_cart": 2, "purchase": 3}
MIN_POINTS = 0.5     # a skipped style fades to almost nothing, but can come back
MAX_POINTS = 10      # so one big favourite doesn't block every other style forever


def make_unit_length(vector):
    return vector / np.linalg.norm(vector)


def starting_tastes(styles, genders):
    """One taste vector per style: the average vector of that style's products for these genders.

    Returns (tastes, styles_used). Styles with no products for these genders, and repeats,
    are skipped, so styles_used[i] is the style of tastes[i]. Both can be empty."""
    tastes = []
    styles_used = []
    for style in styles:
        if style in styles_used:                    # the same style twice counts once
            continue
        total = np.zeros(database.VECTOR_SIZE)
        count = 0
        for product in catalog.products.values():
            if product["style"] == style and product["gender"] in genders:
                total = total + product["vector"]
                count = count + 1
        if count > 0:
            tastes.append(make_unit_length(total / count))
            styles_used.append(style)
    return np.array(tastes, dtype="float32"), styles_used   # tastes: one row per style


def new_user(picked, genders):
    """Tastes, styles and points for a new user: their picked styles plus their neighbours.
    Returns (tastes, styles, points). styles is empty if none of the picked styles exist."""
    wanted = list(picked)
    for style in picked:
        for neighbour in NEIGHBOURS.get(style, []):
            if neighbour not in wanted:
                wanted.append(neighbour)
    tastes, styles = starting_tastes(wanted, genders)
    if not any(style in picked for style in styles):
        return tastes, [], []
    points = []
    for style in styles:
        if style in picked:
            points.append(PICKED_POINTS)
        else:
            points.append(NEIGHBOUR_POINTS)
    return tastes, styles, points


def learn(tastes, styles, points, product_id, action):
    """Learn from one swipe. Returns (tastes, styles, points).

    The taste for the product's style moves towards the product (or away, for a dislike),
    and that style's points go up or down. Liking a product from a style the user doesn't
    have yet adds that style, with a taste that starts at this product."""
    if action not in WEIGHTS:
        print("WARNING unknown action:", action, "- swipe ignored")
        return tastes, styles, points
    if not catalog.has_product(product_id):
        return tastes, styles, points              # product not in our catalogue, skip it

    product_vector = catalog.get_vector(product_id)
    product_style = catalog.products[product_id]["style"]
    styles_line_up = len(styles) == len(tastes) == len(points)

    if styles_line_up and product_style in styles:
        number = styles.index(product_style)
        points = list(points)
        points[number] = min(MAX_POINTS, max(MIN_POINTS, points[number] + POINTS[action]))
    elif styles_line_up and POINTS[action] > 0:
        # a new style they liked: add it, starting from this product
        tastes = np.vstack([tastes, make_unit_length(product_vector)]).astype("float32")
        return tastes, styles + [product_style], list(points) + [POINTS[action]]

    if styles_line_up and product_style in styles:
        closest = styles.index(product_style)          # the taste for this product's own style
    else:
        # Which of the user's tastes is this product most like?
        closest = 0
        best_match = -999
        for number in range(len(tastes)):
            match = np.dot(tastes[number], product_vector)
            if match > best_match:
                best_match = match
                closest = number

    moved = tastes[closest] + WEIGHTS[action] * product_vector
    if np.linalg.norm(moved) < 1e-6:               # almost impossible, but never divide by zero
        return tastes, styles, points

    tastes = tastes.copy()
    tastes[closest] = make_unit_length(moved)
    return tastes, styles, points