"""
Updates the product list and rebuilds the AI numbers.

    python update_products.py new_products.csv

Nothing touches the live folder until the new files are built and checked.
If anything goes wrong the old folder stays and the app keeps working.
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Backend gives us this URL later. Until then we take a file path instead.
BACKEND_URL = os.getenv("BACKEND_URL")

# The columns we cannot work without. Better to stop straight away with a
# clear message than fail halfway through building.
NEEDED = ["product_id", "title", "brand", "gender", "category",
          "subcategory", "style", "price", "primary_image", "is_active"]

LIVE = "data"
NEW = "data_new"
OLD = "data_old"

def get_new_list():
    """Where the new product list comes from."""
    if not BACKEND_URL:
        return sys.argv[1]

    print("fetching products from backend...")
    products = pd.read_csv(BACKEND_URL + "/internal/products.csv")
    products.to_csv("/tmp/new_products.csv", index=False)
    print("got", len(products), "products")
    return "/tmp/new_products.csv"


def check_columns(products):
    """Is this file the shape we expect?"""
    missing = [c for c in NEEDED if c not in products.columns]
    if missing:
        return "missing columns: " + ", ".join(missing)

    if products["product_id"].duplicated().any():
        return "the same product_id appears more than once"

    if products["product_id"].isna().any():
        return "some rows have no product_id"

    return None

def main():
    started = datetime.now()
    new_csv = get_new_list()

    old = pd.read_csv(LIVE + "/products_clean.csv")
    new = pd.read_csv(new_csv)

    problem = check_columns(new)
    if problem:
        return stop(problem, started, 0, 0)

    added = len(set(new["product_id"]) - set(old["product_id"]))
    removed = len(set(old["product_id"]) - set(new["product_id"]))

    print(f"before {len(old)}   after {len(new)}   +{added} -{removed}")

    # A good export never loses half the catalogue. If it has, something
    # broke on the way here, so stop before we wipe out real products.
    if len(new) < len(old) / 2:
        return stop("new list is less than half the old one", started, added, removed)

    # Build the new numbers in a scratch folder, not the live one.
    shutil.rmtree(NEW, ignore_errors=True)
    os.mkdir(NEW)
    shutil.copy(new_csv, NEW + "/products_clean.csv")

    print("building vectors, takes about 10 minutes...")
    env = dict(os.environ, BUILD_INTO=NEW)
    if subprocess.run([sys.executable, "build_embeddings.py"], env=env).returncode:
        return stop("building the vectors failed", started, added, removed)

    # The list and the numbers line up by position, so if they disagree
    # every recommendation would quietly be wrong. Check before swapping.
    vectors = np.load(NEW + "/product_embeddings.npy")
    if len(new) != len(vectors):
        return stop(f"{len(new)} products but {len(vectors)} vectors", started, added, removed)

    if np.isnan(vectors).any():
        return stop("some vectors are broken", started, added, removed)

    # Renaming happens all at once, so the live folder is either the old
    # one or the new one, never half way between.
    shutil.rmtree(OLD, ignore_errors=True)
    os.rename(LIVE, OLD)
    os.rename(NEW, LIVE)

    for keep in ["mock_product_events.csv"]:
        if os.path.exists(OLD + "/" + keep):
            shutil.copy(OLD + "/" + keep, LIVE + "/" + keep)

    write_report("SUCCESS", "", started, added, removed, len(new))
    print("done. restart the service so it loads the new products.")


def stop(reason, started, added, removed):
    """Give up without changing anything."""
    shutil.rmtree(NEW, ignore_errors=True)
    write_report("FAILED", reason, started, added, removed, 0)
    sys.exit(1)


def write_report(status, reason, started, added, removed, total):
    os.makedirs("reports", exist_ok=True)
    minutes = (datetime.now() - started).total_seconds() / 60

    lines = [
        "PRODUCT UPDATE  " + started.strftime("%Y-%m-%d %H:%M"),
        f"added     {added}",
        f"removed   {removed}",
        f"total     {total}",
        f"took      {minutes:.1f} min",
        f"STATUS    {status}",
    ]
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