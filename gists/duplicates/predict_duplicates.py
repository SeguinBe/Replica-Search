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

FEATURE_FILES = ['/scratch/benoit/candidate_duplicates_features.shl',
                 '/home/seguin/Replica-search/candidate_duplicates_close_to_positive.shl']
TRAINING_CSV = '/mnt/cluster-nas/benoit/duplicate_exp.csv'
CLASSIFIER_FILE = '/home/seguin/Replica-search/duplicate_classifier.pkl'
METADATA_CSV = '/home/seguin/Cini-Project/Process-Metadata/output_processed.csv'
OUTPUT_PDF_FILE = '/home/seguin/conflicting_duplicates'
OUTPUT_SCORES_FILE = '/home/seguin/Replica-search/duplicate_scores.pkl'
CONFLICTING_CSV = '/home/seguin/Replica-search/conflicting_stats.csv'

# If we skip prediction, assuming it was done already
PREDICT_SCORES = False

dataset = Dataset('/scratch/benoit/resized_dataset/index.csv')
print(dataset)

dicts = []
for feature_file in FEATURE_FILES:
    d = shelve.open(feature_file, flag='r')
    print("Candidates pairs for duplicates : {}".format(len(d)))
    dicts.append(d)


# In[11]:
if PREDICT_SCORES:

    if TRAINING_CSV:
        training_data = pd.read_csv(TRAINING_CSV)
        all_keys = set()
        for d in dicts:
            all_keys.update(d.keys())
        recall_training = np.mean([k in all_keys for k in training_data[training_data.is_duplicate].key])
        print("Estimation recall (using training samples) : {:.1f}%".format(100 * recall_training))

    # In[15]:
    print("-" * 60)
    class MyPreprocess:
        def transform(self, features):
            features = features.copy()
            features[:, :3] = np.log(features[:, :3])
            features[:, 3:5] = np.sqrt(features[:, 3:5])
            return features

        def fit(self, X, y=None):
            pass


    with open(CLASSIFIER_FILE, 'rb') as f:
        clf_pipeline = pickle.load(f)

    scores = dict()
    for d in dicts:
        for k, (f, boxes) in tqdm(d.items(), desc="Classifying candidates"):
            scores[k] = (clf_pipeline.predict_proba(f[np.newaxis])[0, 1], min([b['h']*b['w'] for b in boxes]))

    write_pickle(scores, OUTPUT_SCORES_FILE)
else:
    scores = read_pickle(OUTPUT_SCORES_FILE)

# In[65]:

import networkx as nx

score_threshold = 0.9
duplicate_links = [k.split('_') for k, (v, _) in scores.items() if v > score_threshold]
g = nx.Graph(duplicate_links)
connected_components = list(nx.connected_components(g))
print("Duplicate links : {} ({:.1f}% of candidates)".format(len(duplicate_links),
                                                            100 * len(duplicate_links) / len(scores)))
nb_images = sum([len(s) for s in connected_components])
nb_objects = len(connected_components)
print("Number of images involved : {} | Number of objects with duplicate images : {}".format(nb_images,
                                                             nb_objects))

# In[66]:
print("-" * 60)

metadata_df = pd.read_csv(METADATA_CSV)
metadata_dict = dict()
for _, r in tqdm(metadata_df.iterrows(), desc='Loading parsed metadata', total=len(metadata_df)):
    metadata_dict['{}_{}'.format(r.Drawer, r.ImageNumber)] = (r.AuthorULAN
                                                              if r.AuthorULAN != 'ulan:500026590'
                                                              else 'ulan:500029319',  # Correction of the Angelico
                                                              r.AuthorModifier,
                                                              r.AuthorOriginal)

@lru_cache(maxsize=None)
def get_metadata_key(img_uid):
    """Returns '105A_319'"""
    return core_server.model.CHO.get_from_image_uid(img_uid).uri.split('/')[-1].split('.')[0]


# In[68]:

from collections import Counter

agreeing_metadata = []
conflicting_metadata = []
conflicts = []
attributed = 0
nb_images_cini = 0
nb_objects_cini = 0
for s in tqdm(connected_components, desc='Checking attribution'):
    ids = []
    nb_images_cini_obj = 0
    for uid in s:
        # author_id = core_server.model.CHO.get_from_image_uid(uid).get_metadata('AuthorId')
        metadata_key = get_metadata_key(uid)
        if metadata_key not in metadata_dict:
            continue
        nb_images_cini += 1
        nb_images_cini_obj += 1
        author_id, author_modifier, _ = metadata_dict[metadata_key]
        # Only considering non-unknown and 'attr' elements
        if isinstance(author_id, str) and 'ulan' in author_id and (
                (not isinstance(author_modifier, str)) or author_modifier in ('?', 'attr')):
            ids.append(author_id)
            attributed += 1
    if len(ids) >= 2:
        if len(set(ids)) == 1:
            agreeing_metadata.append(s)
        else:
            conflicting_metadata.append(s)
            conflicts.append(set(ids))
    if nb_images_cini_obj >= 2:
        nb_objects_cini += 1

conflicts = [frozenset((uid1, uid2)) for c in conflicts for uid1 in c for uid2 in c if uid1 < uid2]
conflicts = Counter(conflicts)

# In[69]:

ulan_id_to_label = dict()
for _id, label in zip(metadata_df.AuthorULAN, metadata_df.AuthorULANLabel):
    ulan_id_to_label[_id] = label

# In[71]:
print("Number of images involved (cini) : {} | Number of objects with duplicate images (cini) : {}".format(nb_images_cini,
                                                             nb_objects_cini))
print("Image with clear attribution \t\t: {} ({:.1f}%)".format(attributed, 100 * attributed / nb_images_cini))
print("Objects with agreeing attribution \t: {} ({:.1f}%)".format(len(agreeing_metadata),
                                                                  100 * len(agreeing_metadata) / nb_objects_cini))
print("Objects with conflicting attribution \t: {} ({:.1f}%)".format(len(conflicting_metadata),
                                                                     100 * len(conflicting_metadata) / nb_objects_cini))
print("Estimation conflicting rate \t: {:.2f}%".format(
    100 * len(conflicting_metadata) / (len(agreeing_metadata) + len(conflicting_metadata))))

# In[72]:

print("-" * 60)
print("Most common attribution conflicts :")

for s, count in conflicts.most_common()[:10]:
    print(*[ulan_id_to_label[_id] for _id in s], '{} artworks'.format(count), sep='\t| ')

data = []
conflict_pairs = []
for s, count in conflicts.most_common():
    _id1, _id2 = tuple(s)
    d = {"id1": _id1, "id2": _id2,
         "label1": ulan_id_to_label[_id1], "label2": ulan_id_to_label[_id2],
         "count": count}
    data.append(d)
df = pd.DataFrame(data=data)
g = nx.Graph()
for i, c in df.iterrows():
    g.add_edge(c.label1, c.label2, {'weight': c['count']})
df.to_csv(CONFLICTING_CSV, index=False)
nx.write_gexf(g, CONFLICTING_CSV.replace('.csv', '.gexf'))

# In[85]:
print("-" * 60)
print("Export to PDF")
involved_images = [uid for g in conflicting_metadata for uid in g]
labels = dict()
links = dict()
for img_uid in tqdm(involved_images):
    cho = core_server.model.CHO.get_from_image_uid(img_uid)
    labels[img_uid] = cho.author
    links[img_uid] = cho.related
dataset.pdf_export_groups(conflicting_metadata, OUTPUT_PDF_FILE, split_files_into=150,
                          labels=labels, links=links)
