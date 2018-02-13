from .base import *
from ..evaluation import Benchmark, TestQuery


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
            if len(targets) == 0:
                continue
            forbidden = set(targets + [q_id])
            for k in forbidden.copy():
                forbidden.update([kk for kk, d in self.connection_graph[k].items()
                                  if d['type'] == ConnectedDataset.LinkType.DUPLICATE])
            # Search candidates
            candidates = [r[0] for r in search_function(q_id, (len(forbidden) + 2*n_negatives))]
            # Filter the ones which are not negatives
            filtered_candidates = [c for c in candidates if c not in forbidden]
            # Extend the list
            if len(filtered_candidates) > 0:
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