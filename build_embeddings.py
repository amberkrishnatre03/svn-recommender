"""
Turns every active product into 512 numbers and stores them in Postgres.

    goes in:   data/products_5k.csv           one row per product
    comes out: products table in Postgres     each active product + its 512 numbers

Run once, from the repo folder:   python build_embeddings.py
Works on your Mac or on the cloud: the database address comes from DATABASE_URL
(.env on your Mac, the cloud's settings on the cloud). Needs internet for the
model download (first time only) and for the product images.
Takes about 15 minutes for 5,000 products.
"""

import os
import numpy as np
import pandas as pd
import requests
import torch
from dotenv import load_dotenv
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

load_dotenv()                     # reads DATABASE_URL from .env, must run before app.database is imported
from app import database

# Which CSV to read. Set PRODUCTS_CSV to use another file, otherwise data/products_5k.csv
CSV_FILE = os.getenv("PRODUCTS_CSV", "data/products_5k.csv")

# How much each part counts in the final vector. Must add up to 1.
IMAGE_WEIGHT = 0.7                # the picture shows the actual look, so it counts more
TEXT_WEIGHT = 0.3                 # the text adds brand, fabric, fit

# 1. LOADS The pretrained AI model from huggingface website. 600 MB. Downloads the first time, saved on disk after that and runs from there
print("loading FashionCLIP...")
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")

# Model needs tensors, This processor converts both images and texts into the multi dimensional array (tensors) format that model needs
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model.eval()       # Tells the model we are only using it for prediction not for training.

products = pd.read_csv(CSV_FILE)                              # loads the csv
print(len(products), "products in", CSV_FILE)

# Only active products with an image link get numbers. Inactive ones are not stored.
products = products[products["is_active"] == True]
products = products[products["primary_image"].notna()]
products = products[products["style"].notna()]      # every product needs a style, the database requires it
products = products.drop_duplicates("product_id")   # the same product twice would stop the save, keep the first one
products = products.reset_index(drop=True)      # row numbers 0,1,2... again, so position = row
products["price"] = pd.to_numeric(products["price"], errors="coerce")   # a price that isn't a number becomes empty, instead of failing the save at the end
total = len(products)           # counts the rows of products
print(total, "active products to embed")


def get_numbers(output):         #this function just converts those tensors into numpy array
    # Older versions of transformers hand back plain numbers.
    # Newer ones wrap them in an object, so we pull them out.
    if not torch.is_tensor(output):
        output = output.pooler_output
    return output.detach().numpy()


def make_unit_length(vectors):
    # Every vector must be exactly 1 unit long so they compare fairly. this function converts it to unit lenght
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)        # lenght of vector 3² + 4² = 25, and √25 = 5
    return vectors / lengths                                        # then we shrink each vector to 1 unit long
                                                                    # each vector element is under 1.0

# ---------- part 1: read the words ----------
print("reading descriptions...")

text_vectors = []                                # an empty pandas list

for start in range(0, total, 32):               # for i in range ( 0,5000 , 32 ) 32 is the step size / why 32 because sending all at once would run out of memory
    lines = products["embedding_text"][start:start + 32].astype(str).tolist()   # turns the pandas result into a plain Python list, which is what the processor expects. astype(str) so a blank text can't crash it
    inputs = processor(text=lines, return_tensors="pt", padding=True, truncation=True)      # runs tokenising for texts inside lines

    with torch.no_grad():                                # no training only evaluation
        batch = model.get_text_features(**inputs)       # model call for text processing ( dict processor)

    text_vectors.append(get_numbers(batch))             # converts the tensor to numpy and adds this batch
    print("  ", start + len(lines), "/", total)         # prints the progess eg 32/5000 then 64/5000 etc

text_vectors = np.vstack(text_vectors)                 # separate blocks of 32 rows. vstack stacks them vertically into one table of total rows by 512 numbers.
text_vectors = make_unit_length(text_vectors)         # shrint to lenght 1


# ---------- part 2: look at the pictures ---------- # downloading the images from primary image column and making vector
print("looking at images...")                        # now we gonna make an empty numpy array
image_vectors = np.zeros((total, text_vectors.shape[1]), dtype="float32")    # total rows and 512 dimensions float type 32 uses half the memory as compared to float 64
image_worked = []                                    # this gonna collect the images that worked

for start in range(0, total, 32):                                     # for i in range (0,total,32)
    urls = products["primary_image"][start:start + 32].tolist()       # Same batching as before, but reading the image URL column.

    images = []                                       # holds the pictures that downloaded successfully in this batch
    positions = []                                     # which row each of those belongs to
                                                  # Two lists because some downloads fail, so the positions do not simply run 0 to 31
    for i, url in enumerate(urls):                # enumurate gives position and value together, to know what row you are in
        try:                                      # i counts 0,1,2 within batch of 32
            answer = requests.get(url, timeout=10, stream=True)        #downloads whatever is in this link using request gives up after 10 seconds so dead connection do not stop this job
            picture = Image.open(answer.raw).convert("RGB")           # answer raw is the download bytes,open turns into picture object then converts to rgb needed for model
            images.append(picture)                                   # keep this picture for the batch
            positions.append(start + i)                               # remembers which row it belongs to
        except Exception:
            pass                       # broken link,wrong file type,- we will use its words instead

    if images:
        try:                                               # one damaged picture can make the model reject the whole batch
            inputs = processor(images=images, return_tensors="pt")   # resizing images into 224*224

            with torch.no_grad():
                batch = model.get_image_features(**inputs)     # gets the images features from the model

            image_vectors[positions] = get_numbers(batch)       # writes the answers into rows listed in positions, every other row keeps its zeroes
            image_worked.extend(positions)                       # adds all batch positions to the running list
        except Exception:
            print("   the model could not read a picture in this batch,", len(images), "products use their words instead")

    print("  ", start + len(urls), "/", total)

print(len(image_worked), "images worked")
print(total - len(image_worked), "images were broken")        # progress of summary


# ---------- part 3: mix words and pictures ----------
# Words tell us the brand and cut. Pictures tell us the actual look.
# Together they describe a product better than either one alone.

image_vectors[image_worked] = make_unit_length(image_vectors[image_worked])   # normalise only the rows that have a picture, the rest stay zero and are not used

final_vectors = text_vectors.copy()      # fall back for broken images
final_vectors[image_worked] = IMAGE_WEIGHT * image_vectors[image_worked] + TEXT_WEIGHT * text_vectors[image_worked]    # 70% picture + 30% words
final_vectors = make_unit_length(final_vectors)     # normalise to unit vector lenght 1


# ---------- part 4: save to Postgres ----------
if len(final_vectors) != len(products):
    print("stopped: got", len(final_vectors), "vectors for", len(products), "products, nothing saved")
else:
    database.create_tables()                                         # makes the products table if it does not exist yet
    database.save_products(products, final_vectors.astype("float32"))  # replaces everything in the table in one go: all saved, or nothing
    print("done")
    print("saved", database.count_products(), "products,", final_vectors.shape[1], "numbers each")        # summarry of products