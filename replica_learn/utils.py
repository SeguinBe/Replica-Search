import tensorflow as tf
from typing import Dict
import pickle
import os
from typing import List
import networkx as nx
import numpy as np


def _load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0


def write_as_summaries(training_directory: str, values_dict: Dict[str, float]):
    writer = tf.summary.FileWriter(os.path.join(training_directory, 'eval'))
    summary = tf.Summary(value=[
        tf.Summary.Value(tag=k, simple_value=v) for k, v in values_dict.items()
    ])
    writer.add_summary(summary, _load_global_step_from_checkpoint_dir(training_directory))
    writer.close()


def read_pickle(filename: str):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def write_pickle(obj, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


class BaseParams:
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        result = cls()
        keys = result.to_dict().keys()
        for k, v in d.items():
            assert k in keys, k
            setattr(result, k, v)
        result.check_params()
        return result

    def check_params(self):
        pass


class ModelParams(BaseParams):
    def __init__(self):
        self.lstm_size = 768
        self.lstm_steps = 3
        self.weight_decay = 1e-5
        self.n_classes = None  # type: int


class TrainingParams(BaseParams):
    def __init__(self):
        self.nb_epochs = 30
        self.evaluate_every_epoch = 1
        self.learning_rate = 5e-6
        self.lr_decay_rate = 0.95
        self.lr_decay_steps = 'auto'  # type: int
        self.batch_size = 32
        self.weighted_loss = False


def separate_links(links: List, proportion: float) -> (List[frozenset], List[frozenset]):
    g = nx.Graph(links)
    components = list(nx.connected_component_subgraphs(g))
    to_be_kept_array = np.random.random_sample(len(components)) < proportion
    output_1, output_2 = [], []
    for component, to_be_kept in zip(components, to_be_kept_array):
        to_be_added_links = [frozenset(e) for e in component.edges()]
        if to_be_kept:
            output_1.extend(to_be_added_links)
        else:
            output_2.extend(to_be_added_links)
    return output_1, output_2


def create_graph_from_edges(links) -> nx.Graph:
    """
    Create a graph where two element of a given group are connected
    :param groups:
    :return:
    """
    g = nx.Graph()
    for l in links:
        g.add_edge(l['img1'], l['img2'], **dict(l))
    return g


def drop_duplicate_clusters(graph: nx.Graph):
    sub_graphs = list(nx.connected_component_subgraphs(graph))
    return nx.union_all([g for g in sub_graphs if any([d['type'] == 'POSITIVE'
                                                       for _, _, d in g.edges(data=True)])])


def separate_graph_into_subgraphs(graph: nx.Graph, proportion=0.8, verbose=True) -> (nx.Graph, nx.Graph):
    """
    Separate a graph into
    :param graph: Incoming
    :param proportion:
    :return: big_sub_graph, small_sub_graph
    """
    assert 0 < proportion < 1
    sub_graphs = list(nx.connected_component_subgraphs(graph))
    print('subgraphs : {}'.format(len(sub_graphs)))
    is_training = np.random.rand(len(sub_graphs)) < proportion
    training_graph = nx.union_all([sub_g for i, sub_g in enumerate(sub_graphs) if is_training[i]])
    validation_graph = nx.union_all([sub_g for i, sub_g in enumerate(sub_graphs) if not is_training[i]])
    if verbose:
        print("Asked proportion : {} | Actual proportion : {} ({}, {})".format(proportion,
                                                                               len(training_graph)/len(graph),
                                                                               len(training_graph),
                                                                               len(validation_graph)))
    return training_graph, validation_graph
