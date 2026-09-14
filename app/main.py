"""
The API the backend calls.

    POST /onboarding   new user picked gender and styles: create their taste, return a feed
    POST /feed         their swipes since the last call: learn from them, return the next feed
    POST /reset        show everything again, keep their taste

Run locally:  python -m uvicorn app.main:app --reload
"""

from dotenv import load_dotenv
load_dotenv()                         # read .env before database.py looks for DATABASE_URL

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app import catalog, database, recommend, vectors

app = FastAPI(title="SVN Recommendations")



@app.exception_handler(HTTPException)
def error_shape(request, problem):
    return JSONResponse({"status": problem.status_code, "message": problem.detail},
                        problem.status_code)


@app.exception_handler(RequestValidationError)
def bad_field(request, problem):
    """A field name we don't know (a typo), or a value of the wrong type.
    Same status/message shape as everything else, every problem listed."""
    messages = []
    for e in problem.errors():
        field = e["loc"][-1]
        if e["type"] == "extra_forbidden":
            messages.append(f"{field} is not a valid field")
        elif e["type"] == "json_invalid":
            messages.append("request body is not valid JSON")
        else:
            messages.append(f"{field} {e['msg'].lower().replace('input should be', 'must be')}")
    return JSONResponse({"status": 400, "message": ", ".join(messages)}, 400)


def required(request, *fields):
    """Raise ONE error naming every field that was not sent or was sent empty,
    e.g. "user_id is Required, gender is Required". "" and "   " count as empty."""
    missing = [f for f in fields
               if getattr(request, f) is None or not str(getattr(request, f)).strip()]
    if missing:
        raise HTTPException(400, ", ".join(f"{f} is Required" for f in missing))


# ---------- what each request must look like ----------
# Every field is optional here so that a missing one still reaches our code,
# where we check it ourselves and answer with a plain message. If a field were
# required, FastAPI would reject the request before our function ran.

class OnboardingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")      # unknown fields are an error
    user_id: str | None = None
    gender: str | None = None
    styles: list[str] | None = None


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid")      # a typo like "actoin" is an error, not ignored
    product_id: str | None = None
    action: str | None = None


class FeedRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str | None = None
    interactions: list[Interaction] | None = None    # can be empty, absent, or null


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str | None = None



# ---------- endpoints ----------

@app.post("/onboarding")
def onboarding(request: OnboardingRequest):
    """Create the user's taste from their gender and styles. Calling again starts them fresh."""
    required(request, "user_id", "gender", "styles")
    if request.gender not in recommend.ALLOWED_GENDER:
        raise HTTPException(400, "gender must be Male, Female or Other")
    if not request.styles:
        raise HTTPException(400, "pick at least 1 style")

    genders = recommend.ALLOWED_GENDER[request.gender]
    tastes, styles, style_scores = vectors.new_user(request.styles, genders)   # picked styles + neighbours
    if len(styles) == 0:
        raise HTTPException(400, "none of those styles exist. valid styles are: "
                            + ", ".join(catalog.all_styles))

    database.save_new_user(request.user_id, request.gender, tastes, styles, style_scores)
    return make_feed(request.user_id, tastes, styles, style_scores, request.gender, [])


@app.post("/feed")
def feed(request: FeedRequest):
    """Learn from the swipes since the last call, then return the next feed."""
    required(request, "user_id")
    interactions = request.interactions or []          # no swipes yet is fine
    for interaction in interactions:
        required(interaction, "product_id", "action")
        if interaction.action not in vectors.ACTION_WEIGHT:
            raise HTTPException(400, "action must be like, dislike or add_to_cart")
        if not catalog.has_product(interaction.product_id):
            raise HTTPException(400, f"product '{interaction.product_id}' does not exist")

    user = database.get_user(request.user_id)
    if user is None:
        raise HTTPException(404, "user not found, call /onboarding first")
    gender, tastes, seen, styles, style_scores = user
    if len(style_scores) != len(styles):                 # users saved before style scores existed
        style_scores = [vectors.PICKED_STYLE_SCORE] * len(styles)

    done = set()
    for interaction in interactions:
        if interaction.product_id in done:         # same product twice in one call counts once
            continue
        done.add(interaction.product_id)
        tastes, styles, style_scores = vectors.learn(tastes, styles, style_scores,
                                               interaction.product_id, interaction.action)

    return make_feed(request.user_id, tastes, styles, style_scores, gender, seen)


@app.post("/reset")
def reset(request: ResetRequest):
    """Forget what they have seen so every product can come back. Their taste stays."""
    required(request, "user_id")

    user = database.get_user(request.user_id)
    if user is None:
        raise HTTPException(404, "user not found")
    gender, tastes, seen, styles, style_scores = user
    if len(style_scores) != len(styles):                 # users saved before style scores existed
        style_scores = [vectors.PICKED_STYLE_SCORE] * len(styles)

    database.clear_seen(request.user_id)
    return make_feed(request.user_id, tastes, styles, style_scores, gender, [])


def make_feed(user_id, tastes, styles, style_scores, gender, seen):
    """Build the next feed, then save their taste, style scores and what they have now seen."""
    products = recommend.build_feed(tastes, styles, style_scores, gender, seen)
    shown = [p["product_id"] for p in products]
    database.update_user(user_id, tastes, styles, style_scores, shown)

    return {
        "products": products,
        "count": len(products),
        "exhausted": len(products) == 0,          # True: they have seen everything, offer /reset
    }