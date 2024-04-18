import sys
import requests
import json
import os
import time

# cfg
subreddit = "FixedPoliticalMemes"
subreddit_folder = "PoliticalMemes"
sort = "controversial"
top_time = "all"
limit = 10000
#  [hot|new|rising|top|controversial] sort
#  [all|year|month|week|day] top time
filename = r"E:\Master\testelast_values.txt"

# if os.path.exists(filename):
#     with open(filename, 'r') as file:
#         last_values = file.read().split(',')
after = None
i =4798
# else:
#     after = None
#     i = 1

while True:
    url = f"https://www.reddit.com/r/{subreddit}/{sort}/.json?count=200"
    if after:
        url += f"&after={after}"
    url += f"&raw_json=1&t={top_time}"
    print(url)
    response = requests.get(url)

    if response.status_code == 429:
        print("Too many requests. Waiting for 5 seconds...")
        time.sleep(5)
        continue

    content = response.json()
    os.makedirs(subreddit, exist_ok=True)

    with open(filename, 'w') as file:
        file.write(f"{after},{i}")

    data = content['data']['children']
    urls = []
    names = []
    ids = []
    for child in data:
        try:
            if "data" in child:
                if "post_hint" in child["data"]:
                    if 'image' in child['data']['post_hint']:
                        urls.append(child['data']['preview']['images'][0]['source']['url'])
                        names.append(child['data']['title'])
                        ids.append(child['data']['id'])
        except Exception as e:
            pass

    a = 0

    for url in urls:
        name = names[a]
        post_id = ids[a]
        ext = url.split('.')[-1].split('?')[0].replace('gif', 'png')
        newname = f"{i}.{ext}"
        print(f"{i}/{limit} : {newname}")
        with open(os.path.join(subreddit_folder, newname), 'wb') as f:
            f.write(requests.get(url).content)
        a += 1
        i += 1

        if i > limit > 0:
            sys.exit(0)

    after = content['data'].get('after', None)
    print(after)

    if not after:
        break
