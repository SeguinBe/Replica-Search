import sys
sys.path.append('/home/seguin/Replica-core/')
#sys.path.append('/home/seguin/Replica-search/')

from core_server import model
from tqdm import tqdm
import pandas as pd
import os
import datetime


OUTPUT_DIR = '/mnt/cluster-nas/benoit'

data_links = []
for l in tqdm(model.VisualLink.nodes):
    uid1, uid2 = [img.uid for img in l.images]
    data_links.append({
            'img1':  uid1, 'img2': uid2, 'uid': l.uid, 'type': l.type, 'annotated': l.annotated
        })

df_links = pd.DataFrame(data_links, columns=['uid', 'img1', 'img2', 'type', 'annotated'])

data_images = []
for img in tqdm(model.Image.nodes):
    if 'WGA' in img.iiif_url:
        source = 'wga'
    elif 'web' in img.iiif_url:
        source = 'web'
    elif 'cini' in img.iiif_url:
        source = 'cini'
    else:
        source = None
    assert source is not None, img.iiif_url
    data_images.append({
            'uid': img.uid, 'added': img.added, 'source': source
        })
df_images = pd.DataFrame(data_images, columns=['uid', 'source', 'added'])

# Save files
os.makedirs(OUTPUT_DIR, exist_ok=True)
now = datetime.datetime.now()
time_suffix = '{:02d}_{:02d}_{:02d}'.format(now.year, now.month, now.day)
df_links.to_pickle(os.path.join(OUTPUT_DIR, 'link_data_{}.pkl'.format(time_suffix)))
df_images.to_pickle(os.path.join(OUTPUT_DIR, 'image_data_{}.pkl'.format(time_suffix)))
