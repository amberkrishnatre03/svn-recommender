"""
Everything about a user's taste: how it starts, and how it changes.

A user has one taste vector per style they picked at onboarding. Keeping
them separate matters: averaging Streetwear and Old Money into one vector
gives a point that is neither, and the feed then feels random.
"""

import numpy as np

from app import catalog

# How much each action moves the user's taste.
# A left swipe pushes gently away, because skips are most of what users do.
WEIGHTS = {
    "left_swipe": -0.10,
    "view": 0.05,
    "right_swipe": 0.30,
    "wishlist": 0.40,
    "add_to_cart": 0.55,
    "purchase": 0.70,
}

# How fast old taste fades. 0.85 keeps 85% of the old vector each time,
# so recent swipes matter more without storing any timestamps.
FADE = 0.85


def make_unit_length(vector):
    """Make the vector exactly 1 unit long, so it compares fairly."""
    return vector / np.linalg.norm(vector)


def build_starting_vectors(styles, gender):
    """New user finished onboarding. One taste vector per style they picked."""
    tastes = []

    for style in styles:
        chosen = catalog.products[
            (catalog.products["style"] == style)
            & (catalog.products["gender"] == gender)
        ]
        if len(chosen) > 0:
            tastes.append(make_unit_length(catalog.vectors[chosen.index].mean(axis=0)))

    if not tastes:
        raise Exception("no products found for " + str(styles) + " " + gender)

    return np.array(tastes, dtype="float32")


def apply_interaction(tastes, product_id, action):
    """User swiped. Move only the taste this product is closest to.

    Liking a polo should sharpen their Old Money taste, not drag their
    Streetwear taste toward polos.
    """
    weight = WEIGHTS.get(action, 0.0)
    if weight == 0:
        return tastes

    product_vector = catalog.get_vector(product_id)

    closest = int(np.argmax(tastes @ product_vector))

    tastes = tastes.copy()
    moved = tastes[closest] * FADE + product_vector * weight
    tastes[closest] = make_unit_length(moved)

    return tastes