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
    "dislike":      -0.05,
    "view":          0.03,
    "like":          0.15,
    "add_to_cart":   0.28,
    "purchase":      0.35,
}

# How fast old taste fades. 0.85 keeps 85% of the old vector each time, 1 means nothing would fade and 0.5 means on every swipe it will change
FADE = 0.85


def make_unit_length(vector):
    """Make the vector exactly 1 unit long, so it compares fairly."""         # we have to make this again as length gets higher after swiping
    return vector / np.linalg.norm(vector)


def build_starting_vectors(styles, gender):
    """New user finished onboarding. One taste vector per style they picked."""      # eg : [streetwear , oversized] , "MALE"
    tastes = []                        # collects one vector per style

    for style in styles:                            # filters the table down to one gender and one style ( STREETWEAR + MEN GIVES ABOUT 300 PRODUCTS)
        chosen = catalog.products[                           # AND MEANS MANDATORY BOTH ARE IMPORTANT AT ONCE AS WOMEN PICKING STREETWEAR SHOULD START FROM WOMEN'S STREETWEAR NOT MENS
            (catalog.products["style"] == style)
            & (catalog.products["gender"] == gender)
        ]
        if len(chosen) > 0:                                          # catalog.vectors[chosen.index] gives that 300 by 512 table ( 300 rows and 512 dimensions for each row)
            tastes.append(make_unit_length(catalog.vectors[chosen.index].mean(axis=0)))        # mean averages down then columns collapsing 300 rows into one row of 512 numbers
                                                                                             # normalise the it back to 1
    if not tastes:                                                                         #
        raise Exception("no products found for " + str(styles) + " " + gender)

    return np.array(tastes, dtype="float32")                    # turns the list of vectors into one numpy table ( TWO ROWS FOR TWO CATEGORIES 3 FOR 3 EACH ROW CONTAINS THE VECTOR FOR THAT CATEGORY)
                                                                # we wil keep seperate use vectors taste for categories
                                                                # taste1( streetwear - ALL PRODUCTS VECTORS AVERAGED...), then taste2(formals ALL CLOTHES PRODUCT VECTOR AVERAGED) STORED INSIDE
                                                             # EG : ROW 1 : 512 COLUMNS ( STREETWEAR) // ROW 2 : 512 COLUMNS ( FORMALS )
def apply_interaction(tastes, product_id, action):                    #called once per swipe
    """User swiped. Move only the taste this product is closest to.

    Liking a polo should sharpen their Old Money taste, not drag their
    Streetwear taste toward polos.
    """
    # An action name we do not recognise almost always means the backend and
    # this service disagree on spelling. Nothing would crash, the swipe would
    # just quietly do nothing, so say so out loud instead.
    if action not in WEIGHTS:                                                    # when we swipe and action name isnt what we recognise
        print("WARNING unknown action:", action, "- swipe ignored")              # then we gonna skip that product in the batch
        return tastes                                                            # rest of the batch still works

    weight = WEIGHTS[action]                        # if new action is added and assigned 0 value
    if weight == 0:                                 # skips the work rather than taking any cation and retuns the weight
        return tastes

    # A product we have never heard of. Usually a test product, or something
    # added to the catalogue after our last rebuild. Skip this one swipe
    # rather than failing the whole request.
    if not catalog.has_product(product_id):
        print("WARNING unknown product_id:", product_id, "- swipe ignored")         # ignores the fake product that dont exist in our catalogue yet
        return tastes                                                               # our 11 stack feed is unaffected

    product_vector = catalog.get_vector(product_id)                                # 512 numbers for the thing they swiped on

    closest = int(np.argmax(tastes @ product_vector))                         # MATRIX MULTIPLICATION ( TASTE IS 2 BY 512 , PRODUCT_VECTOR IS 512 SO RESULT IS 2 NUMBERS )
                                 # this is basically similarity score ( cosine similarity )/ # GIVES ONE SCORE PER TASTE , EG: we had 2 rows and 512 columns for tastes list and product list ( 4120 rows and 512 columns )
    tastes = tastes.copy()                                              # makes tastes copy
    moved = tastes[closest] * FADE + product_vector * weight            #
    tastes[closest] = make_unit_length(moved)                           # since we did addition or subtraction as per weights and taste vector moved
                                                                        # we make it normalised
    return tastes