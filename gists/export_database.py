from tqdm import tqdm
import app
from replica_search import model
import argparse
import pandas as pd
import click
from imageio import imread, imsave
import cv2
import os


@click.group()
def cli():
    pass


def resize(img, max_dim=720):
    if max(img.shape[:2]) < max_dim:
        return img
    if img.shape[0] < img.shape[1]:
        new_size = (max_dim, int(img.shape[0]/img.shape[1]*max_dim))
    else:
        new_size = (int(img.shape[1]/img.shape[0]*max_dim), max_dim)
    return cv2.resize(img, new_size)


def save_tuples_to_csv(uid_path_list, output_file):
    pd.DataFrame(uid_path_list, columns=['uid', 'path']).to_csv(output_file, index=False)


def get_paths():
    return [(img_loc.uid, img_loc.get_image_path())
             for img_loc in tqdm(app.Session().query(model.ImageLocation))]


@cli.command()
@click.argument("output_csv")
def locations_as_csv(output_csv):
    paths = get_paths()
    # Save csv
    save_tuples_to_csv(paths, output_csv)


@cli.command()
@click.option("--max-dim", default=720, help="Maximum dimension for resizing images")
@click.argument("output_dir")
# @click.argument("output_csv")
def images_and_csv(max_dim, output_dir):
    output_csv = os.path.join(output_dir, 'index.csv')
    paths = get_paths()
    output_paths = []
    for uid, path in tqdm(paths, desc="Copying and resizing images"):
        try:
            img = imread(path, pilmode="RGB")
            img = resize(img, max_dim)
            output_folder = os.path.join(output_dir, uid[0], uid[1])
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, uid+'.jpg')
            imsave(output_path, img, quality=90)
            output_paths.append((uid, output_path))
        except Exception as e:
            print(uid, path, e)
    save_tuples_to_csv(output_paths, output_csv)


if __name__ == '__main__':
    cli()
