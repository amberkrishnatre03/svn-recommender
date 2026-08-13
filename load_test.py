"""
Sends many requests at once and measures how fast the API responds.

Start the server first, in another terminal:
    python -m uvicorn app.main:app --workers 4

Then run this.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import requests

URL = "http://127.0.0.1:8000"
USERS = 50          # how many fake users
REQUESTS = 2000    # total requests to send
AT_ONCE = 100       # how many run at the same time

# Change this word to get brand new users. Old users have already seen
# most of the catalogue, which makes the test measure the wrong thing.
PREFIX = "fresh2"


def setup_users():
    """Create the fake users before we start timing."""
    for i in range(USERS):
        requests.post(URL + "/onboarding", json={
            "user_id": "%s_%d" % (PREFIX, i),
            "gender": "Male" if i % 2 == 0 else "Female",
            "styles": ["Streetwear", "Oversized"],
        })
    print("created", USERS, "test users")


def one_request(number):
    """Ask for a feed and return how long it took, in milliseconds."""
    user = "%s_%d" % (PREFIX, number % USERS)

    start = time.time()
    answer = requests.post(URL + "/feed", json={
        "user_id": user,
        "interactions": [],
    })
    taken = (time.time() - start) * 1000

    return taken, answer.status_code


setup_users()
print("sending", REQUESTS, "requests,", AT_ONCE, "at a time...")

start = time.time()

with ThreadPoolExecutor(max_workers=AT_ONCE) as pool:
    results = list(pool.map(one_request, range(REQUESTS)))

total_seconds = time.time() - start

times = sorted(r[0] for r in results)
failed = sum(1 for r in results if r[1] != 200)

print()
print("total time:      %.1f seconds" % total_seconds)
print("requests/second: %.0f" % (REQUESTS / total_seconds))
print("failed:          %d" % failed)
print()
print("fastest:  %5.0f ms" % times[0])
print("average:  %5.0f ms" % (sum(times) / len(times)))
print("median:   %5.0f ms" % times[len(times) // 2])
print("slowest:  %5.0f ms" % times[-1])
print("95%% under: %4.0f ms" % times[int(len(times) * 0.95)])