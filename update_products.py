"""
Updates the product list and rebuilds the AI numbers.

    python update_products.py new_products.csv

Only products that are new, or whose text or image changed, get rebuilt.
Everything else keeps yesterday's numbers, because a vector only depends
on those two things. Deleted products need no work, they are simply not
in the new list so they are never carried across.

Nothing touches the live folder until the new files are built and checked.
If anything goes wrong the old folder stays and the app keeps working.

Run this on a schedule once backend confirms where the CSV comes from.
Linux cron, 4am daily:
    0 4 * * * cd /path/to/svn-recommender && python update_products.py
On AWS this will be a scheduled task set up by the cloud engineer, not
from inside this repo.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Where the updated CSV comes from. First one that is set wins:
#   1. BACKEND_URL   - we fetch it over http
#   2. INCOMING_CSV  - a path the backend team drops the file at
#   3. sys.argv[1]   - a path passed by hand when running manually
BACKEND_URL = os.getenv("BACKEND_URL")                     # either a url
INCOMING_CSV = os.getenv("INCOMING_CSV")              # or a csv from a folder

# The columns we cannot work without. Better to stop straight away with a
# clear message than fail ten minutes into building.
NEEDED = ["product_id", "title", "brand", "gender", "category",
          "subcategory", "style", "primary_image", "is_active",
          "embedding_text"]

# Files that live in the data folder but are not rebuilt by this job.
KEEP = ["mock_product_events.csv"]

# How many numbers each product vector has. Must match build_embeddings.py.
VECTOR_SIZE = 512

LIVE = "data"
NEW = "data_new"
OLD = "data_old"
BUILD = "data_build"


def get_new_list():
    """Where the new product list comes from."""
    if BACKEND_URL:
        print("fetching products from backend...")
        products = pd.read_csv(BACKEND_URL + "/internal/products.csv")
        products.to_csv("/tmp/new_products.csv", index=False)
        print("got", len(products), "products")
        return "/tmp/new_products.csv"

    if INCOMING_CSV:
        print("reading products from", INCOMING_CSV)
        return INCOMING_CSV

    return sys.argv[1]


def clean_rows(products):
    """Drop rows we cannot use, and say how many went.

    Row level problems get cleaned. Structural problems stop the job
    instead, because those mean the export itself is wrong.
    """
    before = len(products)
    dropped = {}

    step = len(products)
    products = products.dropna(subset=["product_id"])
    dropped["no product_id"] = step - len(products)

    step = len(products)
    products = products.drop_duplicates(subset=["product_id"], keep="first")
    dropped["duplicate product_id"] = step - len(products)

    step = len(products)
    products = products.dropna(subset=["embedding_text"])
    dropped["no embedding_text"] = step - len(products)

    step = len(products)
    products = products[products["is_active"] == True]
    dropped["not active"] = step - len(products)

    for reason, count in dropped.items():
        print("dropped", count, "rows:", reason)

    print("kept", len(products), "of", before)

    # Row numbers have to run 0,1,2... with no gaps, because everything
    # below lines vectors up with rows by position.
    return products.reset_index(drop=True), dropped


def check_columns(products):
    """Problems bad enough to stop the whole job.

    These all mean the export itself is wrong, not that a few rows are
    messy. A missing column cannot be cleaned, it has to be fixed upstream.
    """
    missing = [c for c in NEEDED if c not in products.columns]
    if missing:
        return "missing columns: " + ", ".join(missing)

    return None


def check_values(products):
    """Run after cleaning, on the rows we are actually going to keep."""
    if len(products) == 0:
        return "nothing left after cleaning"

    # A lowercase "male" would pass every count check and then give every
    # user an empty feed, because the filter looks for "Men".
    if not products["gender"].isin(["Men", "Women"]).all():
        bad = sorted(set(products["gender"]) - {"Men", "Women"})
        return "gender must be Men or Women, found: " + ", ".join(map(str, bad))

    return None


def plan_rebuild(new):
    """Work out which products already have a good vector.

    A vector only depends on the embedding text and the image. If neither
    changed then yesterday's numbers are still exactly right, so there is
    no point spending time building them again.

    Returns row number -> old vector, for the rows we can reuse. Anything
    not in there has to be built. An empty dict means build everything,
    which is always safe, just slow.
    """
    if not os.path.exists(LIVE + "/product_embeddings.npy"):
        print("no old vectors, building everything")
        return {}

    old = pd.read_csv(LIVE + "/products_clean.csv")
    old_vectors = np.load(LIVE + "/product_embeddings.npy")

    # If the old folder is already inconsistent we cannot trust any of it.
    if len(old) != len(old_vectors):
        print("old list and vectors disagree, building everything")
        return {}

    # A repeated id would mean one product silently claims another one's
    # vector, so do not reuse anything from a file like that.
    if old["product_id"].duplicated().any():
        print("old list has repeated product ids, building everything")
        return {}

    # If the model ever changes size, old vectors are the wrong width and
    # cannot be mixed with new ones.
    if old_vectors.shape[1] != VECTOR_SIZE:
        print("old vectors are the wrong size, building everything")
        return {}

    # What each product looked like last time.
    was = {}
    for i, row in enumerate(old.itertuples()):
        was[row.product_id] = (row.embedding_text, row.primary_image, i)

    reuse = {}
    for i, row in enumerate(new.itertuples()):
        before = was.get(row.product_id)

        if before is None:
            continue                              # new product

        if before[0] != row.embedding_text:
            continue                              # text changed

        if before[1] != row.primary_image:
            continue                              # image changed

        reuse[i] = old_vectors[before[2]]

    print("reusing", len(reuse), "vectors, building", len(new) - len(reuse))
    return reuse


def build_vectors(new, reuse, started, added, removed, dropped):
    """Build whatever cannot be reused, then put the full set in order."""
    to_build = [i for i in range(len(new)) if i not in reuse]

    fresh = {}

    if to_build:
        # Hand build_embeddings.py a small CSV with only these rows. It has
        # no idea it is building a subset, it just embeds what it is given.
        shutil.rmtree(BUILD, ignore_errors=True)
        os.mkdir(BUILD)
        new.iloc[to_build].to_csv(BUILD + "/products_clean.csv", index=False)

        print("building", len(to_build), "vectors...")
        env = dict(os.environ, BUILD_INTO=BUILD)
        failed = subprocess.run([sys.executable, "build_embeddings.py"],
                                env=env).returncode

        if failed:
            return stop("building the vectors failed",
                        started, added, removed, dropped)

        built = np.load(BUILD + "/product_embeddings.npy")
        shutil.rmtree(BUILD, ignore_errors=True)

        if len(built) != len(to_build):
            return stop(f"asked for {len(to_build)} vectors, got {len(built)}",
                        started, added, removed, dropped)

        if built.shape[1] != VECTOR_SIZE:
            return stop(f"vectors are {built.shape[1]} wide, expected {VECTOR_SIZE}",
                        started, added, removed, dropped)

        # built is in the same order as the small CSV we just wrote, so
        # position 0 of it is row to_build[0] of the full list.
        for position, row_number in enumerate(to_build):
            fresh[row_number] = built[position]

    # Row i of this array has to be row i of the new CSV. Everything
    # downstream looks products up by position, so if this is wrong then
    # every recommendation is wrong and nothing errors.
    vectors = np.zeros((len(new), VECTOR_SIZE), dtype="float32")

    for i in range(len(new)):
        if i in fresh:
            vectors[i] = fresh[i]
        else:
            vectors[i] = reuse[i]

    return vectors


def main():
    started = datetime.now()
    new_csv = get_new_list()

    if not os.path.exists(LIVE + "/products_clean.csv"):
        print("no live catalogue yet, this looks like the first run")
        old = pd.DataFrame({"product_id": []})
    else:
        old = pd.read_csv(LIVE + "/products_clean.csv")

    new = pd.read_csv(new_csv)

    problem = check_columns(new)
    if problem:
        return stop(problem, started, 0, 0, {})

    new, dropped = clean_rows(new)

    problem = check_values(new)
    if problem:
        return stop(problem, started, 0, 0, dropped)

    added = len(set(new["product_id"]) - set(old["product_id"]))
    removed = len(set(old["product_id"]) - set(new["product_id"]))

    print(f"before {len(old)}   after {len(new)}   +{added} -{removed}")

    # A good export never loses half the catalogue. If it has, something
    # broke on the way here, so stop before we wipe out real products.
    if len(old) and len(new) < len(old) / 2:
        return stop("new list is less than half the old one",
                    started, added, removed, dropped)

    reuse = plan_rebuild(new)
    vectors = build_vectors(new, reuse, started, added, removed, dropped)

    # Everything from here writes into a scratch folder, never the live one.
    shutil.rmtree(NEW, ignore_errors=True)
    os.mkdir(NEW)

    # Write the cleaned rows, not the file we were given. Copying the
    # original would throw away everything clean_rows just did.
    new.to_csv(NEW + "/products_clean.csv", index=False)
    np.save(NEW + "/product_embeddings.npy", vectors)
    np.save(NEW + "/product_ids.npy", new["product_id"].values)

    # Bring across anything this job does not rebuild, before the swap, so
    # the live folder is never missing a file it needs.
    for name in KEEP:
        if os.path.exists(LIVE + "/" + name):
            shutil.copy(LIVE + "/" + name, NEW + "/" + name)

    # The list and the numbers line up by position, so if they disagree
    # every recommendation would quietly be wrong. Check before swapping.
    if len(new) != len(vectors):
        return stop(f"{len(new)} products but {len(vectors)} vectors",
                    started, added, removed, dropped)

    if np.isnan(vectors).any():
        return stop("some vectors are broken", started, added, removed, dropped)

    # Same length is not enough, the order has to match too. A shuffled
    # array passes every other check here and is just as wrong.
    saved_ids = np.load(NEW + "/product_ids.npy", allow_pickle=True)
    if not (new["product_id"].values == saved_ids).all():
        return stop("vector order does not match the products",
                    started, added, removed, dropped)

    # Renaming happens all at once, so the live folder is either the old
    # one or the new one, never half way between.
    shutil.rmtree(OLD, ignore_errors=True)
    if os.path.exists(LIVE):
        os.rename(LIVE, OLD)
    os.rename(NEW, LIVE)

    write_report("SUCCESS", "", started, added, removed, len(new), dropped)
    print("done. restart the service so it loads the new products.")
    print("to undo:  rm -rf data && mv data_old data")


def stop(reason, started, added, removed, dropped):
    """Give up without changing anything."""
    shutil.rmtree(NEW, ignore_errors=True)
    shutil.rmtree(BUILD, ignore_errors=True)
    write_report("FAILED", reason, started, added, removed, 0, dropped)
    sys.exit(1)


def write_report(status, reason, started, added, removed, total, dropped):
    os.makedirs("reports", exist_ok=True)
    minutes = (datetime.now() - started).total_seconds() / 60

    lines = [
        "PRODUCT UPDATE  " + started.strftime("%Y-%m-%d %H:%M"),
        f"added     {added}",
        f"removed   {removed}",
        f"total     {total}",
        f"took      {minutes:.1f} min",
    ]

    for reason_text, count in dropped.items():
        if count:
            lines.append(f"dropped   {count} rows: {reason_text}")

    lines.append(f"STATUS    {status}")

    if reason:
        lines.append(f"reason    {reason}")
        lines.append("nothing changed, live data untouched")

    text = "\n".join(lines)
    name = "reports/update_" + started.strftime("%Y-%m-%d_%H%M") + ".txt"
    open(name, "w").write(text + "\n")

    print()
    print(text)


if __name__ == "__main__":
    main()