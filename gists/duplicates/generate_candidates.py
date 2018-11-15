import shelve
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
# Needs Replica-core in path
import core_server
from replica_learn.dataset import Dataset
from functools import lru_cache
from replica_learn.utils import write_pickle, read_pickle
import requests
import tqdm
import pandas as pd

# Number of closest to selected images to retrieve as candidates
top_n = 20+1
OUTPUT_CSV = '/home/seguin/duplicate_candidates_close_to_positive.csv'
# Search server URL
SEARCH_SERVER = "http://iccluster052.iccluster.epfl.ch:5001"

all_links = core_server.model.VisualLink.nodes.all()

img_uids = list({img.uid for l in all_links if l.type == "POSITIVE" for img in l.images})


candidates = []
for img_uid in tqdm.tqdm(img_uids):
    try:
        r = requests.post('{}/api/search'.format(SEARCH_SERVER),
                          json={'positive_image_uids': [img_uid], 'rerank': True}).json()
        r_uids = [rr['uid'] for rr in r['results'][:top_n] if rr['uid'] != img_uid]
        candidates.extend([frozenset((img_uid, r_uid)) for r_uid in r_uids])
    except Exception as e:
        print("Error with {}".format(img_uid))
candidates = list(set(candidates))

data = [tuple(s) for s in candidates]
pd.DataFrame(data, columns=['img_uid1', 'img_uid2']).to_csv(OUTPUT_CSV, index=False)
