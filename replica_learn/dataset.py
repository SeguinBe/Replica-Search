from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json
from PIL import Image

try:
    from pylatex import Document, Section, Tabular, StandAloneGraphic, NoEscape, NewLine, HFill
except:
    pass
import networkx as nx
from tqdm import tqdm
import pandas as pd
from .evaluation import Benchmark, TestQuery


class Dataset:
    def __init__(self, path_dict: Union[Dict[str, str], str]):
        if isinstance(path_dict, str):
            df = pd.read_csv(path_dict)
            self.path_dict = {r.uid: r.path for i, r in df.iterrows()}
        else:
            self.path_dict = path_dict

    def __repr__(self):
        return 'Dataset {} elements total'.format(len(self.path_dict))

    def plot_pair(self, uuid1, uuid2, label):
        plt.subplot(1, 2, 1)
        self.plot_img(uuid1)
        plt.subplot(1, 2, 2)
        self.plot_img(uuid2)
        plt.title('Label : {}'.format(label))

    def plot_img(self, uuid: str, box=None):
        arr = np.asanyarray(Image.open(self.get_image_path(uuid)))
        plt.imshow(arr)
        current_axis = plt.gca()
        if box is not None:
            h, w = arr.shape[:2]
            box = [box[0]*h, box[1]*w, box[2]*h, box[3]*w]
            current_axis.add_patch(
                Rectangle(
                    (box[1], box[0]),
                    box[3], box[2],
                    edgecolor="red", fill=False)
            )

    def plot_imgs(self, uuids: List[str]):
        plt.figure(figsize=(15, 15))
        nb_results = len(uuids)
        grid_size = np.ceil(np.sqrt(nb_results))
        for i, uuid in enumerate(uuids):
            plt.subplot(grid_size, grid_size, i+1)
            self.plot_img(uuid)

    def plot_query(self, query, results, region=None):
        plt.figure(figsize=(15, 15))
        nb_results = len(results)
        grid_size = np.ceil(np.sqrt(nb_results+1))
        plt.subplot(grid_size, grid_size, 1)
        self.plot_img(query, region)

        for i, res in enumerate(results):
            if region is not None:
                t, s, r = res
            else:
                t, s = res
                r = None
            plt.subplot(grid_size, grid_size, i+2)
            self.plot_img(t, r)
            plt.title("#{}: {:.3f}".format(i+1, s))

    def get_image_path(self, uuid: str):
        return self.path_dict[uuid]

    def pdf_export_of_pairs(self, filename: str, pairs: List[Tuple[str, str, float]]):
        pos_pairs = [(i, p) for i, p in enumerate(pairs) if p[2] > 0]
        neg_pairs = [(i, p) for i, p in enumerate(pairs) if p[2] == 0]
        geometry_options = {"rmargin": "1cm", "lmargin": "1cm"}
        doc = Document(geometry_options=geometry_options)

        def draw_pairs_tables(pairs_to_be_drawn):
            n = 0
            for i, p in pairs_to_be_drawn:
                uuid1, uuid2 = p[:2]
                with doc.create(Tabular('|c|c|')) as table:
                    table.add_hline()
                    table.add_row((StandAloneGraphic(self.path_dict[uuid1],
                                                     image_options=NoEscape(r'width=0.12\textwidth')),
                                   StandAloneGraphic(self.path_dict[uuid2],
                                                     image_options=NoEscape(r'width=0.12\textwidth'))))
                    table.add_hline()
                    table.add_row((i, '' if len(p) < 4 else 'Score:{:.4f}'.format(p[3])))
                    table.add_hline()
                n += 1
                if n % 3 == 0:
                    doc.append(NewLine())
                else:
                    doc.append(HFill())

        with doc.create(Section('Positive pairs')):
            draw_pairs_tables(pos_pairs)
        with doc.create(Section('Negative pairs')):
            draw_pairs_tables(neg_pairs)

        doc.generate_pdf(filename)

    @staticmethod
    def export_pairs(json_filename: str, pairs: List[Tuple]):
        """
        NB: might be used for List[Tuple4] (uuid1, uuid2, label, score)
        :param json_filename:
        :param pairs:
        :return:
        """
        with open(json_filename, 'w') as f:
            json.dump(pairs, f, indent=4)

    def save_examples_to_csv(self, csv_filename: str, examples: List[Tuple]):
        """
        Save pairs or triplets of training examples by converting their uids to their absolute path
        :param csv_filename:
        :param examples:
        :return:
        """
        converted_elements = [tuple(self.get_image_path(p) if isinstance(p, str) else p for p in t)
                              for t in examples]
        df = pd.DataFrame(data=converted_elements)
        df.to_csv(csv_filename, header=False, index=False)

    @staticmethod
    def import_pairs(json_filename: str) -> List[Tuple]:
        """
        NB: might be used for List[Tuple4] (uuid1, uuid2, label, score)
        :param json_filename:
        :return:
        """
        with open(json_filename, 'r') as f:
            return [tuple(r) for r in json.load(f)]


class ConnectedDataset(Dataset):
    class LinkType:
        POSITIVE = 'POSITIVE'
        NEGATIVE = 'NEGATIVE'
        DUPLICATE = 'DUPLICATE'
        ALL_TYPES = {DUPLICATE, NEGATIVE, POSITIVE}

    def __init__(self, path_dict: Union[Dict[str, str], str], connection_graph: nx.Graph, drop_unknown_nodes=False):
        super().__init__(path_dict=path_dict)
        self.connection_graph = connection_graph.copy()

        # Checks
        assert isinstance(self.connection_graph, nx.Graph)
        # Remove unknown edge type
        for uid1, uid2, type in self.connection_graph.edges(data='type'):
            if type not in self.LinkType.ALL_TYPES:
                self.connection_graph.remove_edge(uid1, uid2)

        # Remove disconnected nodes
        for n in self.connection_graph.nodes():
            if len(self.connection_graph[n]) == 0:
                self.connection_graph.remove_node(n)

        if drop_unknown_nodes:
            dropped = 0
            for uuid in self.connection_graph.nodes():
                if uuid not in self.path_dict.keys():
                    dropped += 1
                    self.connection_graph.remove_node(uuid)
            print('Dropped {} elements'.format(dropped))

        for uuid in self.connection_graph:
            assert uuid in self.path_dict.keys(), uuid

    def __repr__(self):
        return '{} | {} connections between {} elements'.format(super().__repr__(),
                                                                len(self.connection_graph.edges()),
                                                                len(self.connection_graph))

    def sample_positive_pairs(self, p=1.0) -> List[Tuple[str, str, float]]:
        assert 0 < p <= 1.0, "p should be a probability"
        all_positives = []
        for n1, n2, t in self.connection_graph.edges_iter(data='type'):
            if t == self.LinkType.POSITIVE:
                all_positives.append((n1, n2, 1.0))
        sampled_positives = []
        for i in np.random.choice(len(all_positives), int(len(all_positives) * p), replace=False):
            sampled_positives.append(all_positives[i])
        return sampled_positives

    def sample_simple_positive_pairs(self, n_pairs) -> List[Tuple[str, str, float]]:
        all_uuids = np.array(list(self.path_dict.keys()))
        sampled_positives = []
        for i in range(n_pairs):
            n1 = np.random.choice(all_uuids)
            sampled_positives.append((n1, n1, 1.))
        return sampled_positives

    def sample_random_negative_pairs(self, n_pairs) -> List[Tuple[str, str, float]]:
        all_uuids = np.array(list(self.path_dict.keys()))
        sampled_negatives = []
        for i in range(n_pairs):
            n1, n2 = np.random.choice(all_uuids, 2, replace=False)
            if not self.connection_graph.has_edge(n1, n2):
                sampled_negatives.append((n1, n2, 0.))
        return sampled_negatives

    def sample_hard_negative_pairs(self, search_function, n_pairs) -> List[Tuple[str, str, float]]:

        queries = self.connection_graph.nodes()
        n_negatives_per_query = [len(a) for a in np.array_split(np.arange(n_pairs), len(queries))]
        assert sum(n_negatives_per_query) == n_pairs
        assert len(queries) == len(n_negatives_per_query)
        sampled_negatives = []
        # For every element annotated in the graph
        for q_id, n_negatives in zip(tqdm(queries, desc='Mining hard negatives'), n_negatives_per_query):
            # Forbidden targets for the pair
            forbidden = set(self.connection_graph.neighbors(q_id) + [q_id])
            for k in forbidden.copy():
                forbidden.update([kk for kk, d in self.connection_graph[k].items()
                                  if d['type'] == ConnectedDataset.LinkType.DUPLICATE])
            # Search candidates
            candidates = [r[0] for r in search_function(q_id, (len(forbidden) + 10) * 3)]
            # Filter the ones which are not negatives
            filtered_candidates = [c for c in candidates if c not in forbidden]
            # Extend the list
            for n_id in np.random.choice(np.array(filtered_candidates), n_negatives, replace=False):
                sampled_negatives.append((q_id, n_id, 0.))
        return sampled_negatives

    def sample_triplets(self, search_function, n_triplets) -> List[Tuple[str, str, str]]:
        queries = [n for n in self.connection_graph.nodes() if len([kk for kk, d in self.connection_graph[n].items()
                                                                    if d['type'] == ConnectedDataset.LinkType.POSITIVE])>0]
        n_triplets_per_query = [len(a) for a in np.array_split(np.arange(n_triplets), len(queries))]
        assert sum(n_triplets_per_query) == n_triplets
        assert len(queries) == len(n_triplets_per_query)
        triplets = []
        # For every element annotated in the graph
        for q_id, n_negatives in zip(tqdm(queries, desc='Mining triplets'), n_triplets_per_query):
            # Forbidden targets for the pair
            targets = [kk for kk, d in self.connection_graph[q_id].items()
                       if d['type'] == ConnectedDataset.LinkType.POSITIVE]
            forbidden = set(targets + [q_id])
            for k in forbidden.copy():
                forbidden.update([kk for kk, d in self.connection_graph[k].items()
                                  if d['type'] == ConnectedDataset.LinkType.DUPLICATE])
            # Search candidates
            candidates = [r[0] for r in search_function(q_id, (len(forbidden) + 2*n_negatives))]
            # Filter the ones which are not negatives
            filtered_candidates = [c for c in candidates if c not in forbidden]
            # Extend the list
            if len(targets) > 0 and len(filtered_candidates) > 0:
                for t_id, n_id in zip(np.random.choice(np.array(targets), n_negatives, replace=True),
                                      np.random.choice(np.array(filtered_candidates), n_negatives, replace=False)):
                    triplets.append((q_id, t_id, n_id))
        return triplets


class BenchmarkGenerator:
    def __init__(self):
        pass

    def generate_benchmark(self, dataset: ConnectedDataset) -> Benchmark:
        benchmark = Benchmark()

        for n in dataset.connection_graph.nodes():
            targets, to_ignore = [], []
            for k, d in dataset.connection_graph[n].items():
                if d['type'] == ConnectedDataset.LinkType.POSITIVE:
                    targets.append(k)
                    to_ignore.extend([kk for kk, d in dataset.connection_graph[k].items()
                                      if d['type'] == ConnectedDataset.LinkType.DUPLICATE])
                elif d['type'] == ConnectedDataset.LinkType.DUPLICATE:
                    to_ignore.append(k)
            if len(targets) > 0:
                benchmark.add_query(TestQuery(n, targets, to_ignore, weight=1.))

        return benchmark


class PairGenerator:
    def __init__(self, negative_positive_ratio, hard_negative_ratio=0, additional_positive_ratio=0):
        self.all_negatives_ratio = negative_positive_ratio
        self.hard_negatives_ratio = hard_negative_ratio
        self.augmented_positives_ratio = additional_positive_ratio
        self.generated_positives_ratio = 0

        assert 0. <= self.hard_negatives_ratio <= 1.
        assert 0. <= self.generated_positives_ratio <= 1.

    def generate_training_pairs(self, dataset: ConnectedDataset, search_function=None):
        all_pairs = []

        # Annotated positives
        annotated_positives = dataset.sample_positive_pairs(p=1.0)
        all_pairs.extend(annotated_positives)

        # Simple positives
        n_simple_positives = int(
            len(annotated_positives) * (1 - self.generated_positives_ratio) * self.augmented_positives_ratio)
        simple_positives = dataset.sample_simple_positive_pairs(n_simple_positives)
        all_pairs.extend(simple_positives)

        # Generated positives
        # if self.generated_positives_ratio > 0.:
        #    n_generated_positives = int(
        #        len(annotated_positives) * self.generated_positives_ratio * self.augmented_positives_ratio)
        #    generated_positives = dataset.sample_generated_positive_pairs(n_generated_positives)
        #    all_pairs.extend(generated_positives)

        # Hard negatives
        if self.hard_negatives_ratio > 0.:
            assert search_function is not None, "Need a search function if we want to mine hard negatives"
            n_hard_negatives = int(len(annotated_positives) * self.hard_negatives_ratio * self.all_negatives_ratio)
            hard_negatives = dataset.sample_hard_negative_pairs(search_function, n_pairs=n_hard_negatives)
            all_pairs.extend(hard_negatives)

        # Random negatives
        n_random_negatives = int(len(annotated_positives) * (1 - self.hard_negatives_ratio) * self.all_negatives_ratio)
        random_negatives = dataset.sample_random_negative_pairs(n_pairs=n_random_negatives)
        all_pairs.extend(random_negatives)

        # Remove potential duplicates?
        all_pairs = [(uuid1, uuid2, l) if uuid1 < uuid2 else (uuid2, uuid1, l) for uuid1, uuid2, l in all_pairs]
        all_pairs = list({tuple(row) for row in all_pairs})

        return all_pairs
