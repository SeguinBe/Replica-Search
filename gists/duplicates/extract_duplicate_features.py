import click
from replica_learn import dataset
from replica_search import duplicates
import pandas as pd
from tqdm import tqdm
import shelve


@click.command()
@click.option('-d', 'dataset_csv')
@click.option('-i', 'pairs_csv')
@click.option('-o', 'output_file')
def extract_duplicate_features(dataset_csv, pairs_csv, output_file):
    d = dataset.Dataset(path_dict=dataset_csv)
    pairs = pd.read_csv(pairs_csv)
    with shelve.open(output_file) as output_dict:
        for i, (img_uid1, img_uid2) in tqdm(pairs.iterrows(), total=len(pairs)):
            k = "{}_{}".format(min(img_uid1, img_uid2), max(img_uid1, img_uid2))
            # Skip if already computed
            if k in output_dict:
                continue
            f_vector, boxes = duplicates.get_duplicate_features(img_uid1, img_uid2, return_boxes=True, dataset=d)

            output_dict[k] = (f_vector, boxes)

            if i % 1000 == 0:
                output_dict.sync()


if __name__ == '__main__':
    extract_duplicate_features()


# python gists/extract_duplicate_features.py -d /scratch/benoit/resized_dataset/index.csv -i candidate_duplicates.csv -o candidate_duplicates_features.shl

# export PYTHONPATH=`pwd`; nice python gists/extract_duplicate_features.py -d /scratch/benoit/resized_dataset/index.csv -i ~/duplicate_candidates_close_to_positive.csv -o candidate_duplicates_close_to_positive.shl