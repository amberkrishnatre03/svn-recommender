"""
Manages the user's taste as vectors — one per style they care about.

Each taste vector points in the direction of products the user likes for that style.
When they swipe, the vector moves towards liked products and away from disliked ones.
A style's score decides how many of the 14 personalised cards it gets.
"""

import numpy as np

from app import catalog, database

# --- tuning knobs ---

# How far one swipe moves the taste vector towards/away from the product.
# Bigger = faster learning but also faster drift if the user swipes randomly.
ACTION_WEIGHT = {
    "dislike":     -0.08,
    "view":         0.01,
    "like":         0.15,
    "add_to_cart":  0.25,
    "purchase":     0.35,
}

# How much each action raises or lowers a style's score, which is what decides
# its share of the personalised cards.
STYLE_WEIGHT_CHANGE = {
    "dislike":    -0.3,
    "view":        0,
    "like":        1,
    "add_to_cart": 2,
    "purchase":    3,
}

MIN_STYLE_SCORE = 0.5   # a disliked style fades low but can always come back
MAX_STYLE_SCORE = 10    # stops one style from eating the entire feed

# The score a style starts with: high if the user picked it, low if it only came
# along as a neighbour.
PICKED_STYLE_SCORE    = 5
NEIGHBOUR_STYLE_SCORE = 0.5

# Neighbour styles are deliberately NOT listed here. catalog.neighbours works
# them out by comparing each style's average product vector, so a style added to
# the database tomorrow gets sensible neighbours with no code change at all.


# --- internal helpers ---

def unit(vector):
    """Scale a vector to length 1 so dot products equal cosine similarity."""
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return vector   # nearly zero vector, leave it alone
    return vector / norm


def style_centroid(style, genders):
    """Mean vector of all products in this style for these genders. None if no products found."""
    total = np.zeros(database.VECTOR_SIZE)
    count = 0
    for product in catalog.products.values():
        if product["style"] == style and product["gender"] in genders:
            total += product["vector"]
            count += 1
    if count == 0:
        return None
    return unit(total / count)


# --- public API ---

def new_user(picked_styles, genders):
    """Build the starting taste vectors, style names and style scores for a new user.

    picked_styles: the styles they chose at onboarding (e.g. ["Oversized", "Casual"])
    genders: product genders they can see (e.g. ["Women"])

    Returns (tastes, styles, style_scores).
    tastes[i] is the taste vector for styles[i].
    Returns (empty, [], []) if none of the picked styles have products.
    """
    # include neighbours so cold-start users see related styles too.
    # catalog.neighbours is built from the catalogue, never from a list in code.
    wanted = list(picked_styles)
    for style in picked_styles:
        for neighbour in catalog.neighbours.get(style, []):
            if neighbour not in wanted:
                wanted.append(neighbour)

    tastes  = []
    styles  = []
    style_scores  = []
    for style in wanted:
        if style in styles:
            continue
        vec = style_centroid(style, genders)
        if vec is None:
            continue
        tastes.append(vec)
        styles.append(style)
        style_scores.append(PICKED_STYLE_SCORE if style in picked_styles else NEIGHBOUR_STYLE_SCORE)

    # only return something useful if at least one picked style exists
    if not any(s in picked_styles for s in styles):
        return np.array([], dtype="float32"), [], []

    return np.array(tastes, dtype="float32"), styles, style_scores


def learn(tastes, styles, style_scores, product_id, action):
    """Update the taste vectors and style scores from one swipe.

    - Moves the taste vector for the product's style towards (or away from) the product.
    - Raises or lowers that style's score.
    - If the user likes a product from a style they don't have yet, adds it.
    """
    if action not in ACTION_WEIGHT:
        print("WARNING: unknown action", action, "— ignored")
        return tastes, styles, style_scores

    if not catalog.has_product(product_id):
        return tastes, styles, style_scores

    product_vector = catalog.get_vector(product_id)
    product_style  = catalog.products[product_id]["style"]
    shift          = ACTION_WEIGHT[action]
    score_change   = STYLE_WEIGHT_CHANGE[action]

    styles = list(styles)
    style_scores = list(style_scores)

    if product_style in styles:
        i = styles.index(product_style)

        # move the taste vector
        moved = tastes[i] + shift * product_vector
        tastes = tastes.copy()
        tastes[i] = unit(moved)

        # adjust the style's score
        style_scores[i] = min(MAX_STYLE_SCORE, max(MIN_STYLE_SCORE, style_scores[i] + score_change))

    elif score_change > 0:
        # user liked something from a new style — add it
        new_taste = unit(product_vector)
        tastes = np.vstack([tastes, new_taste]).astype("float32")
        styles.append(product_style)
        style_scores.append(score_change)

    return tastes, styles, style_scores