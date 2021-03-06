from .base import *
from ..evaluation import Benchmark, TestQuery
from collections import defaultdict
from copy import deepcopy
import networkx as nx


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
        to_be_removed = []
        for uid1, uid2, type in self.connection_graph.edges(data='type'):
            if type not in self.LinkType.ALL_TYPES:
                to_be_removed.append((uid1, uid2))
        for t in to_be_removed:
            self.connection_graph.remove_edge(*t)

        # Remove disconnected nodes
        to_be_removed = []
        for n in self.connection_graph.nodes():
            if len(self.connection_graph[n]) == 0:
                to_be_removed.append(n)
        for n in to_be_removed:
            self.connection_graph.remove_node(n)

        if drop_unknown_nodes:
            to_be_removed = []
            for uuid in self.connection_graph.nodes():
                if uuid not in self.path_dict.keys():
                    to_be_removed.append(uuid)
            for n in to_be_removed:
                self.connection_graph.remove_node(n)
            print('Dropped {} elements'.format(len(to_be_removed)))

        for uuid in self.connection_graph:
            assert uuid in self.path_dict.keys(), uuid

        physical_graph = self.connection_graph.edge_subgraph([(u, v)
                                                              for u, v, d in self.connection_graph.edges(data='type')
                                                              if d == 'DUPLICATE'])
        self.physical_closure = {n: {n} for n in self.connection_graph.nodes()}
        for p_g in nx.connected_components(physical_graph):
            for n in p_g:
                self.physical_closure[n] = p_g
        self.to_be_ignored_dict = defaultdict(set)
        # add the visual->physical connections
        visual_graph = self.connection_graph.edge_subgraph([(u, v)
                                                            for u, v, d in self.connection_graph.edges(data='type')
                                                            if d == 'POSITIVE'])
        for n in visual_graph.nodes():
            neighbors = visual_graph.neighbors(n)
            self.to_be_ignored_dict[n].update(self.physical_closure[n])
            for nn in neighbors:
                self.to_be_ignored_dict[n].update(self.physical_closure[nn])

        # Expand to physical->visual->physical
        tmp = deepcopy(self.to_be_ignored_dict)
        for n, to_be_ignored_set in self.to_be_ignored_dict.items():
            for nn in self.physical_closure[n]:
                to_be_ignored_set.update(tmp[nn])

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

    def sample_triplets(self, search_function, n_triplets, margin=0.0) -> List[Tuple[str, str, str]]:
        queries = [n for n in self.connection_graph.nodes() if len([kk for kk, d in self.connection_graph[n].items()
                                                                    if d[
                                                                        'type'] == ConnectedDataset.LinkType.POSITIVE]) > 0]
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
            forbidden = self.to_be_ignored_dict[q_id]
            # Search candidates
            candidates = search_function(q_id, max(len(forbidden) + 2 * n_negatives, 200))
            candidate_uids, candidate_scores = [], []
            targets_scores = {t_id: 0 for t_id in targets}
            for uid, s in candidates:
                if uid in targets:
                    targets_scores[uid] = s
                elif uid in forbidden:
                    pass
                else:
                    candidate_uids.append(uid)
                    candidate_scores.append(s)

            # Mine hard negatives
            candidate_uids = np.array(candidate_uids)
            candidate_scores = np.array(candidate_scores)
            query_triplets = []
            for t_id, t_s in targets_scores.items():
                t_candidates = candidate_uids[candidate_scores > t_s-2.0*margin]
                if len(t_candidates) > 0:
                    for n_id in np.random.choice(t_candidates, min(n_negatives, len(t_candidates)), replace=False):
                        query_triplets.append((q_id, t_id, n_id))
            if len(query_triplets) > n_negatives:
                choices = np.random.choice(len(query_triplets), n_negatives)
                query_triplets = [query_triplets[i] for i in choices]
            triplets.extend(query_triplets)
        return triplets
            # candidates = [r[0] for r in search_function(q_id, (len(forbidden) + 2 * n_negatives))]
            # running_candidates = []
            # for c in candidates:
            #     if c in forbidden:
            #         continue
            #     elif c in targets:
            #         for n_id in np.random.choice(np.array(running_candidates),
            #                                      min(n_negatives, len(running_candidates)), replace=False):
            #             triplets.append((q_id, c, n_id))
            #     else:
            #         running_candidates.append(c)
            # # Filter the ones which are not negatives
            # filtered_candidates = [c for c in candidates if c not in forbidden]
            # # Extend the list
            # if len(filtered_candidates) > 0:
            #     for t_id, n_id in zip(np.random.choice(np.array(targets), n_negatives, replace=True),
            #                           np.random.choice(np.array(filtered_candidates), n_negatives, replace=False)):
            #         triplets.append((q_id, t_id, n_id))



class BenchmarkGenerator:
    def __init__(self):
        pass

    def generate_benchmark(self, dataset: ConnectedDataset) -> Benchmark:
        benchmark = Benchmark()

        for n in dataset.connection_graph.nodes():
            targets, to_ignore = [], dataset.to_be_ignored_dict[n]
            for k, d in dataset.connection_graph[n].items():
                if d['type'] == ConnectedDataset.LinkType.POSITIVE:
                    targets.append(k)
            if len(targets) > 0:
                benchmark.add_query(TestQuery(n, targets, to_ignore=to_ignore, weight=1.))

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
