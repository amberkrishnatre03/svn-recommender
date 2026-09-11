"""
A user's taste, as vectors.

A user has one taste vector per style they picked at onboarding. Keeping them
separate means someone who likes both Formal and Streetwear gets both, instead
of one average that sits in between and matches neither.
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


def make_unit_length(vector):
    return vector / np.linalg.norm(vector)


def starting_tastes(styles, genders):
    """One taste vector per style: the average vector of that style's products for these genders.
    Styles with no products are skipped, so the result can be empty."""
    tastes = []
    for style in styles:
        total = np.zeros(database.VECTOR_SIZE)
        count = 0
        for product in catalog.products.values():
            if product["style"] == style and product["gender"] in genders:
                total = total + product["vector"]
                count = count + 1
        if count > 0:
            tastes.append(make_unit_length(total / count))
    return np.array(tastes, dtype="float32")      # a table: one row per style


def learn(tastes, product_id, action):
    """Move the user's closest taste towards the product (or away from it, for a dislike)."""
    if action not in WEIGHTS:
        print("WARNING unknown action:", action, "- swipe ignored")
        return tastes
    if not catalog.has_product(product_id):
        return tastes                              # product not in our catalogue, skip it

    product_vector = catalog.get_vector(product_id)

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
        return tastes

    tastes = tastes.copy()
    tastes[closest] = make_unit_length(moved)
    return tastes