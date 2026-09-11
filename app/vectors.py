"""
A user's taste, as vectors.

A user has one taste vector per style they picked at onboarding. Keeping them
separate means someone who likes both Formal and Streetwear gets both, instead
of one average that sits in between and matches neither.
"""

import numpy as np

from app import catalog

# How far one action moves the user's taste towards (or away from) a product.
# Unknown action names are ignored, so the spelling must match what the app sends.
WEIGHTS = {
    "dislike":      -0.05,
    "view":          0.03,
    "like":          0.15,
    "add_to_cart":   0.28,
    "purchase":      0.35,
}


def make_unit_length(vector):
    return vector / np.linalg.norm(vector)


def starting_tastes(styles, genders):
    """One taste vector per style: the average of that style's products for these genders.

    Returns an empty table if none of the styles exist, so the caller can say so.
    """
    tastes = []
    for style in styles:
        rows = np.where((catalog.style_of == style) & np.isin(catalog.gender_of, genders))[0]
        if len(rows) > 0:
            tastes.append(make_unit_length(catalog.vectors[rows].mean(axis=0)))

    return np.array(tastes, dtype="float32").reshape(-1, catalog.vectors.shape[1])


def learn(tastes, product_id, action):
    """Move the closest taste vector towards the product (or away, for a dislike)."""
    if action not in WEIGHTS:
        print("WARNING unknown action:", action, "- swipe ignored")
        return tastes
    if not catalog.has_product(product_id):
        return tastes                               # product not in our catalogue, skip it

    product = catalog.get_vector(product_id)
    closest = int(np.argmax(tastes @ product))      # the taste this product is most like

    moved = tastes[closest] + WEIGHTS[action] * product
    if np.linalg.norm(moved) < 1e-6:                # almost impossible, but never divide by zero
        return tastes

    tastes = tastes.copy()
    tastes[closest] = make_unit_length(moved)
    return tastes