from replica_search.index import IntegralImagesIndex
from replica_core import model
from replica_learn.utils import write_pickle, read_pickle
from tqdm import tqdm
# For the authentication
import core_server

"""
# Old stuff
# For duplicates use a default non-learned metric
# Get a search index
index = IntegralImagesIndex('/tmp/tensorboard3/test_wga/index_0.hdf5')
# Get the bot user for annotations
user = model.User.nodes.get(username='duplicate_bot')
# Find the duplicates
results = index.find_duplicates(max_threshold=0.065)
print("Found {} connections".format(len(results)))
# Optional save them to pdf to check them (must be not too many though)
# dataset.pdf_export_of_pairs('/home/seguin/duplicate_pairs', [(p[0], p[1], 0, d) for p in results])


# Create proposals or directly DUPLICATE annotations in the graph
from tqdm import tqdm
for r in tqdm(results):
    img1 = model.Image.nodes.get(uid=r[0])
    img2 = model.Image.nodes.get(uid=r[1])
    link = model.VisualLink.create_proposal(img1, img2, user)
    # link.annotate(user, model.VisualLink.Type.DUPLICATE)
"""


scores_raw = read_pickle('/home/seguin/Replica-search/duplicate_scores.pkl')
scores_raw = {frozenset(k.split('_')): v for k, v in scores_raw.items()}

THRESHOLD = 0.9
scores = {k:v for k, v in scores_raw.items() if v[0] > 0.9}

user = model.User.nodes.get(username='duplicate_bot')

for (img_uid1, img_uid2), (score, spatial_spread) in tqdm(scores.items()):
    link = model.VisualLink.get_from_images(img_uid1, img_uid2)
    if link is None:
        img1 = model.Image.nodes.get(uid=img_uid1)
        img2 = model.Image.nodes.get(uid=img_uid2)
        link = model.VisualLink.create_proposal(img1, img2, user)
        link.annotate(user, model.VisualLink.Type.DUPLICATE)
    link.prediction_score = score
    link.spatial_spread = spatial_spread
    link.save()
