"""
Turns every product into 512 numbers.
Run this once. It takes about 10 minutes.
"""

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# Load FashionCLIP. The first run downloads it (about 600 MB).
print("loading FashionCLIP...")
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
model.eval()

products = pd.read_csv("data/products_clean.csv")
total = len(products)
print(total, "products")


def get_numbers(output):
    # Older versions of transformers hand back plain numbers.
    # Newer ones wrap them in an object, so we pull them out.
    if not torch.is_tensor(output):
        output = output.pooler_output
    return output.detach().numpy()


def make_unit_length(vectors):
    # Every vector must be exactly 1 unit long so they compare fairly.
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / lengths


# ---------- part 1: read the words ----------
print("reading descriptions...")

text_vectors = []

for start in range(0, total, 32):
    lines = products["embedding_text"][start:start + 32].tolist()
    inputs = processor(text=lines, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        batch = model.get_text_features(**inputs)

    text_vectors.append(get_numbers(batch))
    print("  ", start + len(lines), "/", total)

text_vectors = np.vstack(text_vectors)
text_vectors = make_unit_length(text_vectors)


# ---------- part 2: look at the pictures ----------
print("looking at images...")
image_vectors = np.zeros((total, text_vectors.shape[1]), dtype="float32")
image_worked = []

for start in range(0, total, 32):
    urls = products["primary_image"][start:start + 32].tolist()

    images = []
    positions = []

    for i, url in enumerate(urls):
        try:
            answer = requests.get(url, timeout=10, stream=True)
            picture = Image.open(answer.raw).convert("RGB")
            images.append(picture)
            positions.append(start + i)
        except Exception:
            pass  # broken link, we will use its words instead

    if images:
        inputs = processor(images=images, return_tensors="pt")

        with torch.no_grad():
            batch = model.get_image_features(**inputs)

        image_vectors[positions] = get_numbers(batch)
        image_worked.extend(positions)

    print("  ", start + len(urls), "/", total)

print(len(image_worked), "images worked")
print(total - len(image_worked), "images were broken")


# ---------- part 3: mix words and pictures ----------
# Words tell us the brand and cut. Pictures tell us the actual look.
# Together they describe a product better than either one alone.

image_vectors = make_unit_length(image_vectors)

final_vectors = text_vectors.copy()
final_vectors[image_worked] = text_vectors[image_worked] + image_vectors[image_worked]
final_vectors = make_unit_length(final_vectors)


# ---------- part 4: save ----------
np.save("data/product_embeddings.npy", final_vectors.astype("float32"))
np.save("data/product_ids.npy", products["product_id"].values)

print("done")
print("saved", final_vectors.shape[0], "products,", final_vectors.shape[1], "numbers each")