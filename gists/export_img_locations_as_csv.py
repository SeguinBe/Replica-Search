from tqdm import tqdm
import app
from replica_search import model
import argparse
import pandas as pd


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output-file", required=True, help="Output file")
    args = vars(ap.parse_args())

    OUTPUT_FILE = args['output_file']

    paths = [(img_loc.uid, img_loc.get_image_path()) for img_loc in tqdm(app.Session().query(model.ImageLocation))]
    pd.DataFrame(paths, columns=['uid', 'path']).to_csv(OUTPUT_FILE, index=False)
