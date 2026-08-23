"""
The API your backend calls.

    POST /onboarding   new user finished signup, give them their first feed
    POST /feed         send us their swipes, get the next feed
    POST /reset        they have seen everything, show it all again
    GET  /             the swipe page for testing by hand
"""
# python -m uvicorn app.main:app --reload

from pathlib import Path
from app import catalog, database, recommend, vectors
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict


app = FastAPI(title="SVN Recommendations")

database.create_table()


class OnboardingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    gender: str
    styles: list[str]


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str
    action: str


class FeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str
    interactions: list[Interaction]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str


@app.post("/onboarding")
def onboarding(request: OnboardingRequest):
    """New user picked their styles. Build their taste and give them a feed.

    Calling this again for an existing user starts them completely fresh.
    """
    if request.gender not in recommend.ALLOWED_GENDER:
        raise HTTPException(400, "gender must be Male or Female")

    if not request.styles:
        raise HTTPException(400, "styles cannot be empty")

    wanted = "Men" if request.gender == "Male" else "Women"

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

    database.save_new_user(request.user_id, request.gender, tastes)

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
    for one in request.interactions:
        tastes = vectors.apply_interaction(tastes, one.product_id, one.action)

    return build_and_save(request.user_id, tastes, gender, seen)


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


def catalog_styles():
    """The style names that actually exist in the catalogue right now."""
    from app import catalog
    return catalog.products["style"].dropna().unique().tolist()


def build_and_save(user_id, tastes, gender, seen):
    """Make a feed, then remember what we showed so it never repeats."""
    products = recommend.build_feed(tastes, gender, seen)

    shown = [p["product_id"] for p in products]
    database.update_user(user_id, tastes, shown)

    return {"products": products, "count": len(products)}

@app.get("/trending")
def trending(gender: str, window: str = "7d", limit: int = 20,
             brand: str = None, style: str = None, category: str = None):
    """Most interacted-with products, worked out by the trending job.

    brand takes a comma separated list, so the app can send several at once.

    An empty list is a normal answer, not an error. A narrow window on a
    quiet day genuinely has nothing to show.
    """
    if gender not in recommend.ALLOWED_GENDER:
        raise HTTPException(400, "gender must be Male or Female")

    if window not in ("1d", "7d", "30d", "most_liked"):
        raise HTTPException(400, "window must be 1d, 7d, 30d or most_liked")
    brands = brand.split(",") if brand else None
    allowed = recommend.ALLOWED_GENDER[gender]

    products = []
    for product_id in database.get_trending(window, limit * 5, brands, style, category):
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

    return {"products": products, "count": len(products), "window": window}

@app.get("/")
def tester():
    """A simple page for swiping through the feed by hand."""
    return FileResponse(Path(__file__).parent / "tester.html")