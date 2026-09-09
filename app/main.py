"""
The API your backend calls.

    POST /onboarding   new user finished signup, give them their first feed
    POST /feed         send us their swipes, get the next feed
    POST /reset        they have seen everything, show it all again
    GET  /trending     most interacted with products right now
    GET  /style-dna    style breakdown for the profile page
    GET  /             the swipe page for testing by hand
"""
# python -m uvicorn app.main:app --reload

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from app import catalog, database, recommend, style_dna, vectors       # loads all these files 

app = FastAPI(title="SVN Recommendations")
                                # prints table ready
database.create_table()        # to load the database.py file and run the create table if not exists function

# If a window has nothing in it, try a wider one. A quiet 24 hours is        # this works when 1 day data not found 7 day data not found ..
# normal, especially early on, and an empty screen looks broken.              # dictionary it is fall back functions so if 1day is not found it falls back to 7 day etc...
WIDER_WINDOW = {"1d": "7d", "7d": "30d", "30d": None, "most_liked": None}

                                                                 # model config = confiddict is reserved name pydantic looks for forbid means..
# this is for onboarding call from backend/frontend
class OnboardingRequest(BaseModel):                               # class x (base model) means x is base model that inherits all the pydantic validation settings
    model_config = ConfigDict(extra="forbid")                      # forbid means -if data is not in this form then it rejects
                                                         # if data is in wrong shape then it gives 422 automatically
    user_id: str                                               # user id must exist and it should be in string type
    gender: str                                               # same
    styles: list[str]                                    # styles should be there in list form as it has multiple styles


class Interaction(BaseModel):                                # interactions should be in this form ( this isnt for request api its the interaction column
    model_config = ConfigDict(extra="ignore")               # that is inside feed request that consists of user id and interactions and inside interactions it should have this

    product_id: str                       # "product id" : "product123"
    action: str                           # "action" = "right swipe"


class FeedRequest(BaseModel):                                   # this is for feed call from backend
    model_config = ConfigDict(extra="ignore")

    user_id: str
    interactions: list[Interaction]                      # list of objects defined above ( see class interaction ) see above
                                                        # example : {    "user_id": "usr_1",
                                                       # "interactions": [{"product_id": "ABC", "action": "right_swipe"}] }

class ResetRequest(BaseModel):                            # this is for feed reset call that removes all the seen products but keeps their taste vector same
    model_config = ConfigDict(extra="forbid")              # only user id is needed

    user_id: str


@app.post("/onboarding")                                              # backend/frontend calls this domain link/onboarding
def onboarding(request: OnboardingRequest):                                      #
    """New user picked their styles. Build their taste vector and give them a feed.
    Calling this again for an existing user starts them completely fresh. basically account reset .or preferance reset
    """
    if request.gender not in recommend.ALLOWED_GENDER:
        raise HTTPException(400, "gender must be Male or Female")
# {"user_id": "test_1", "gender": "Male", "styles": ["Streetwear", "Oversized"]}
    if not request.styles:
        raise HTTPException(400, "styles cannot be empty")

    wanted = recommend.ALLOWED_GENDER[request.gender][0]       # wanted = Men

    # If none of the style names match our catalogue, that is bad input from
    # the caller, not a broken service. Say so with a 400 and list what we
    # actually accept, instead of letting it become a confusing 500.
    try:
        tastes = vectors.build_starting_vectors(request.styles, wanted)
    except Exception:
        valid = sorted(catalog_styles())
        raise HTTPException(
            400,
            "none of those styles exist. valid styles are: " + ", ".join(valid),
        )

    database.save_new_user(request.user_id, request.gender, tastes, request.styles)

    return build_and_save(request.user_id, tastes, request.gender, [])


@app.post("/feed")
def feed(request: FeedRequest):
    """Apply the swipes since last time, then give the next feed."""
    user = database.get_user(request.user_id)

    if user is None:
        raise HTTPException(404, "user not found, call /onboarding first")

    gender, tastes, seen = user

    # Move their taste based on what they did. Unknown products and unknown
    # action names are skipped inside apply_interaction, so one bad row can
    # never take down the whole request.
    acted_on = []

    for i in request.interactions:
        if i.product_id in acted_on:
            continue
        acted_on.append(i.product_id)
        tastes = vectors.apply_interaction(tastes, i.product_id, i.action)

    return build_and_save(request.user_id, tastes, gender, seen + acted_on, acted_on)


@app.post("/reset")
def reset(request: ResetRequest):
    """Show this user everything again, without changing their taste.

    Only the seen list is emptied. To reset their taste as well,
    call /onboarding again instead.
    """
    user = database.get_user(request.user_id)

    if user is None:
        raise HTTPException(404, "user not found")

    gender, tastes, seen = user
    database.clear_seen(request.user_id)

    return build_and_save(request.user_id, tastes, gender, [])


@app.get("/trending")
def trending(gender: str, window: str = "7d", limit: int = 20,
             brand: str = None, style: str = None, category: str = None):
    """Most interacted with products, worked out by the trending job.

    brand takes a comma separated list, so the app can send several at once.

    If the window asked for is empty we quietly try a wider one, and the
    "showing" field says which window the products actually came from.
    """
    if gender not in recommend.ALLOWED_GENDER:
        raise HTTPException(400, "gender must be Male or Female")

    if window not in WIDER_WINDOW:
        raise HTTPException(400, "window must be 1d, 7d, 30d or most_liked")

    brands = brand.split(",") if brand else None
    allowed = recommend.ALLOWED_GENDER[gender]

    # Keep widening until we find something, or run out of windows to try.
    showing = window
    product_ids = []

    while showing and not product_ids:
        product_ids = database.get_trending(
            showing, limit * 5, brands, style, category
        )
        if not product_ids:
            showing = WIDER_WINDOW[showing]

    products = []
    for product_id in product_ids:
        if not catalog.has_product(product_id):
            continue

        row = catalog.row_of_product[product_id]

        # Trending is worked out across everyone, so filter to this user's
        # gender here rather than storing a separate list per gender.
        if catalog.products["gender"].iloc[row] not in allowed:
            continue

        product = catalog.get_product(row)
        product["source"] = "trending"
        products.append(product)

        if len(products) == limit:
            break

    return {"products": products, "count": len(products),
            "window": window, "showing": showing}


@app.get("/style-dna")
def get_style_dna(user_id: str):
    """The style breakdown for the profile page slider.

    based_on is how many interactions the percentages come from. Below
    about 10 the numbers are noise, so the app should hide the widget.
    An empty list is a normal answer, not an error.
    """
    return style_dna.build(user_id)


def catalog_styles():
    """The style names that actually exist in the catalogue right now."""
    return catalog.products["style"].dropna().unique().tolist()


def build_and_save(user_id, tastes, gender, seen, acted_on=None):
    """Make a feed, then remember what the user actually swiped on."""
    products = recommend.build_feed(tastes, gender, seen)

    if acted_on is None:
        acted_on = []

    database.update_user(user_id, tastes, acted_on)

    return {
        "products": products,
        "count": len(products),
        "exhausted": len(products) == 0,
    }


@app.get("/")
def tester():
    """A simple page for swiping through the feed by hand."""
    return FileResponse(Path(__file__).parent / "tester.html")