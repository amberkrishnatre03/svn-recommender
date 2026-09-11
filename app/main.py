"""
The API the backend calls.

    POST /onboarding   new user picked gender and styles: create their taste, return a feed
    POST /feed         their swipes since the last call: learn from them, return the next feed
    POST /reset        show everything again, keep their taste
    GET  /             a page for testing the feed by hand

Run locally:  python -m uvicorn app.main:app --reload
"""

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()                         # read .env before database.py looks for DATABASE_URL

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from app import catalog, database, recommend, vectors

app = FastAPI(title="SVN Recommendations")


# ---------- what each request must look like ----------
# Wrong shape -> FastAPI answers 422 automatically, our code never runs.

class OnboardingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")      # unknown fields are an error
    user_id: str
    gender: str
    styles: list[str]


class Interaction(BaseModel):
    model_config = ConfigDict(extra="ignore")      # extra fields from the backend are fine
    product_id: str
    action: str


class FeedRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    user_id: str
    interactions: list[Interaction] = []           # can be empty


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str


# ---------- endpoints ----------

@app.post("/onboarding")
def onboarding(request: OnboardingRequest):
    """Create the user's taste from their gender and styles. Calling again starts them fresh."""
    if request.gender not in recommend.ALLOWED_GENDER:
        raise HTTPException(400, "gender must be Male, Female or Other")
    if not request.styles:
        raise HTTPException(400, "pick at least 1 style")

    genders = recommend.ALLOWED_GENDER[request.gender]
    tastes, styles, points = vectors.new_user(request.styles, genders)   # picked styles + neighbours
    if len(styles) == 0:
        raise HTTPException(400, "none of those styles exist. valid styles are: "
                            + ", ".join(catalog.all_styles))

    database.save_new_user(request.user_id, request.gender, tastes, styles, points)
    return make_feed(request.user_id, tastes, styles, points, request.gender, [])


@app.post("/feed")
def feed(request: FeedRequest):
    """Learn from the swipes since the last call, then return the next feed."""
    user = database.get_user(request.user_id)
    if user is None:
        raise HTTPException(404, "user not found, call /onboarding first")
    gender, tastes, seen, styles, points = user
    if len(points) != len(styles):                 # users saved before points existed
        points = [vectors.PICKED_POINTS] * len(styles)

    done = set()
    for interaction in request.interactions:
        if interaction.product_id in done:         # same product twice in one call counts once
            continue
        done.add(interaction.product_id)
        tastes, styles, points = vectors.learn(tastes, styles, points,
                                               interaction.product_id, interaction.action)

    return make_feed(request.user_id, tastes, styles, points, gender, seen)


@app.post("/reset")
def reset(request: ResetRequest):
    """Forget what they have seen so every product can come back. Their taste stays."""
    user = database.get_user(request.user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    gender, tastes, seen, styles, points = user
    if len(points) != len(styles):                 # users saved before points existed
        points = [vectors.PICKED_POINTS] * len(styles)

    database.clear_seen(request.user_id)
    return make_feed(request.user_id, tastes, styles, points, gender, [])


@app.get("/")
def tester():
    return FileResponse(Path(__file__).parent / "tester.html")


def make_feed(user_id, tastes, styles, points, gender, seen):
    """Build the next feed, then save their taste, points and everything they have now seen."""
    products = recommend.build_feed(tastes, styles, points, gender, seen)
    shown = [p["product_id"] for p in products]
    database.update_user(user_id, tastes, styles, points, seen + shown)

    return {
        "products": products,
        "count": len(products),
        "exhausted": len(products) == 0,          # True: they have seen everything, offer /reset
    }