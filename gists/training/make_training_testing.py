import sys
sys.path.append('/home/seguin/Replica-search/')
import replica_learn, replica_search
from replica_learn import utils, dataset, evaluation
import pandas as pd
import networkx as nx
import os


df_links = utils.read_pickle('/home/seguin/cluster-nas/benoit/link_data_2018_08_02.pkl')
df_images = utils.read_pickle('/home/seguin/cluster-nas/benoit/image_data_2018_08_02.pkl')
df_path = pd.read_csv('/home/seguin/resized_dataset/index.csv')


g = utils.create_graph_from_edges(df_links.to_dict('records'))
# Remove useless duplicate only parts
def drop_duplicate_clusters(graph: nx.Graph):
    sub_graphs = list(nx.connected_component_subgraphs(graph))
    return nx.union_all([g for g in sub_graphs if any([d['type'] == 'POSITIVE' for _, _, d in g.edges(data=True)])])
g_filtered = drop_duplicate_clusters(g)


df_images = df_images[df_images.source != 'cini']
# Warning: some details might be present, and not in the index
img_set = set(df_images.uid).union(set(g_filtered.nodes()))

df_path = df_path[[v in img_set for v in df_path.uid.values]]
path_dict = {d.uid: d.path for _, d in df_path.iterrows()}

full_dataset = dataset.ConnectedDataset(path_dict, g_filtered)
print("Full dataset:", full_dataset)


# Separate the total graph in two
graph_1, graph_2 = utils.separate_graph_into_subgraphs(g_filtered, 0.5)
dataset_1 = dataset.ConnectedDataset(path_dict, graph_1)
dataset_2 = dataset.ConnectedDataset(path_dict, graph_2)

graph_1_training, graph_1_validation = utils.separate_graph_into_subgraphs(graph_1, 0.8)
dataset_1_training = dataset.ConnectedDataset(path_dict, graph_1_training)
dataset_1_validation = dataset.ConnectedDataset(path_dict, graph_1_validation)
graph_2_training, graph_2_validation = utils.separate_graph_into_subgraphs(graph_2, 0.8)
dataset_2_training = dataset.ConnectedDataset(path_dict, graph_2_training)
dataset_2_validation = dataset.ConnectedDataset(path_dict, graph_2_validation)

bench_gen = dataset.BenchmarkGenerator()
benchmark_1 = bench_gen.generate_benchmark(dataset_1)
benchmark_2 = bench_gen.generate_benchmark(dataset_2)
benchmark_1_validation = bench_gen.generate_benchmark(dataset_1_validation)
benchmark_2_validation = bench_gen.generate_benchmark(dataset_2_validation)

OUTPUT_DIR = '/home/seguin/experiment_data_wga'
os.makedirs(OUTPUT_DIR, exist_ok=True)
utils.write_pickle(full_dataset, os.path.join(OUTPUT_DIR, 'dataset_full.pkl'))
utils.write_pickle(dataset_1, os.path.join(OUTPUT_DIR, 'dataset_1.pkl'))
utils.write_pickle(benchmark_1, os.path.join(OUTPUT_DIR, 'benchmark_1.pkl'))
utils.write_pickle(benchmark_1_validation, os.path.join(OUTPUT_DIR, 'benchmark_validation_1.pkl'))
utils.write_pickle(dataset_1_training, os.path.join(OUTPUT_DIR, 'dataset_1_training.pkl'))
utils.write_pickle(dataset_1_validation, os.path.join(OUTPUT_DIR, 'dataset_1_validation.pkl'))

utils.write_pickle(dataset_2, os.path.join(OUTPUT_DIR, 'dataset_2.pkl'))
utils.write_pickle(benchmark_2, os.path.join(OUTPUT_DIR, 'benchmark_2.pkl'))
utils.write_pickle(benchmark_2_validation, os.path.join(OUTPUT_DIR, 'benchmark_validation_2.pkl'))
utils.write_pickle(dataset_2_training, os.path.join(OUTPUT_DIR, 'dataset_2_training.pkl'))
utils.write_pickle(dataset_2_validation, os.path.join(OUTPUT_DIR, 'dataset_2_validation.pkl'))
