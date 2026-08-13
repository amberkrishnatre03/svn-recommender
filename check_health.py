"""Checks the feed is actually correct, not just that it responds."""

from app import recommend, vectors

user = vectors.build_starting_vectors(["Streetwear", "Oversized", "Old Money"], "Men")
seen = []
problems = 0

for round_number in range(1, 6):
    feed = recommend.build_feed(user, "Male", seen)

    if len(feed) != 11:
        print("FAIL feed", round_number, "has", len(feed), "products, expected 11")
        problems += 1

    ids = [p["product_id"] for p in feed]
    if len(ids) != len(set(ids)):
        print("FAIL feed", round_number, "contains a duplicate")
        problems += 1

    if any(p["product_id"] in seen for p in feed):
        print("FAIL feed", round_number, "repeats an old product")
        problems += 1

    for p in feed:
        user = vectors.apply_interaction(user, p["product_id"], "right_swipe")
        seen.append(p["product_id"])

print("problems found:", problems)
print("products shown:", len(seen))