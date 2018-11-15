import shelve
import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
# Needs Replica-core in path
import core_server
from replica_learn.dataset import Dataset
from functools import lru_cache
from replica_core import model

DATASET_FILE = '/scratch/benoit/resized_dataset/index.csv'
OUTPUT_PDF_FILE = '/home/seguin/replica_dataset'

links = model.VisualLink.nodes.filter(type='POSITIVE').all()
link_times = [l.added for l in links]

links = [tuple(img.uid for img in l.images) for l in links]

import networkx as nx

g = nx.Graph(links)
connected_components = list(nx.connected_components(g))
print("Visual links : {}".format(len(links)))
nb_images = sum([len(s) for s in connected_components])
nb_connected_components = len(connected_components)
print("Number of images involved : {} | Number of connected components with duplicate images : {}".format(nb_images,
                                                                                                          nb_connected_components))

# In[66]:
print("-" * 60)

dataset = Dataset(DATASET_FILE)
print(dataset)

# In[85]:
print("-" * 60)
print("Export to PDF")
involved_images = [uid for g in connected_components for uid in g]
labels = dict()
links = dict()
for img_uid in tqdm(involved_images):
    cho = core_server.model.CHO.get_from_image_uid(img_uid)
    labels[img_uid] = cho.author
    links[img_uid] = cho.related
dataset.pdf_export_groups(connected_components, OUTPUT_PDF_FILE, split_files_into=150,
                          labels=labels, links=links)

