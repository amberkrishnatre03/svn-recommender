"""
Turns every product into 512 numbers so we can compare them later.

    goes in:   data/products_clean.csv        4120 rows
    comes out: data/product_embeddings.npy    4120 rows x 512 numbers
               data/product_ids.npy           4120 product ids

Run once. Takes about 10 minutes.
"""

import os
import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# 1. LOADS The pretrained AI model from huggingface website. 600 MB. Downloads the first time, saved on disk after that and runs from there
print("loading FashionCLIP...")
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")

# Model needs tensors, This processor converts both images and texts into the multi dimensional array (tensors) format that model needs
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model.eval()       # Tells the model we are only using it for prediction not for training.

# update_products.py sets this so we can build into a scratch folder
# first and only swap it in once the new files have been checked.
FOLDER = os.getenv("BUILD_INTO", "data")          # if enviroment variable called build into exists then use that path if it doesnt
                                                  # if it doesnt then use data     ( eg build into

products = pd.read_csv(FOLDER + "/products_clean.csv")        # for now data/products_clean.csv # loads the csv
total = len(products)           # counts the rows of products
print(total, "products")        # returns the total lenght of product


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

for start in range(0, total, 32):               # for i in range ( 0,4120 , 32 ) 4120 is (total) variable , 32 is the step size / why 32 because sending all 4120 at once would run out of memory
    lines = products["embedding_text"][start:start + 32].tolist()         # turns the pandas result into a plain Python list, which is what the processor expects.
    inputs = processor(text=lines, return_tensors="pt", padding=True, truncation=True)      # runs tokenising for texts inside lines

    with torch.no_grad():                                # no training only evaluation
        batch = model.get_text_features(**inputs)       # model call for text processing ( dict processor)

    text_vectors.append(get_numbers(batch))             # converts the tensor to numpy and adds this batch
    print("  ", start + len(lines), "/", total)         # prints the progess eg 32/4140 then 64/4140 etc

text_vectors = np.vstack(text_vectors)                 # We now have 129 separate blocks of 32 rows. vstack stacks them vertically into one table of 4,120 rows by 512 numbers.
text_vectors = make_unit_length(text_vectors)         # shrint to lenght 1


# ---------- part 2: look at the pictures ---------- # downloading the images from primary image column and making vector
print("looking at images...")                        # now we gonna make an empty numpy array
image_vectors = np.zeros((total, text_vectors.shape[1]), dtype="float32")    # total is 4120 and shape is 512 dimensions float type 32 uses half the memory as compared to float 64
image_worked = []                                    # this gonna collect the images that worked

for start in range(0, total, 32):                                     # for i in range (0,4120,32)
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
        inputs = processor(images=images, return_tensors="pt")   # resizing images into 224*224

        with torch.no_grad():
            batch = model.get_image_features(**inputs)     # gets the images features from the model

        image_vectors[positions] = get_numbers(batch)       # writes the answers into rows listed in positions, every other row keeps its zeroes
        image_worked.extend(positions)                       # adds all batch positions to the running list

    print("  ", start + len(urls), "/", total)

print(len(image_worked), "images worked")
print(total - len(image_worked), "images were broken")        # progress of summary


# ---------- part 3: mix words and pictures ----------
# Words tell us the brand and cut. Pictures tell us the actual look.
# Together they describe a product better than either one alone.

image_vectors = make_unit_length(image_vectors)         # normalise to unit vector lenght 1

final_vectors = text_vectors.copy()      # fall back for broken images
final_vectors[image_worked] = text_vectors[image_worked] + image_vectors[image_worked]    # together these describe the product better
final_vectors = make_unit_length(final_vectors)     # normalise to unit vector lenght 1


# ---------- part 4: save ----------
np.save(FOLDER + "/product_embeddings.npy", final_vectors.astype("float32"))      # saves the numpy table to disk for product embeddings from final vectors that consist of both images and texts and if eiother one dont work it still contains atleast one of those features so it never croashes
np.save(FOLDER + "/product_ids.npy", products["product_id"].values)               # saves the numpy table of product_id from product table csv

print("done")
print("saved", final_vectors.shape[0], "products,", final_vectors.shape[1], "numbers each")        # summarry of products