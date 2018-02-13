import argparse
import pandas as pd
import numpy as np
import scipy.io as sio
from glob import glob
from collections import defaultdict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-dir", required=True, help="Input directory")
    args = vars(ap.parse_args())

    INPUT_MAT = args['input_dir'] + '/rijksgt.mat'
    IMG_DIR = args['input_dir'] + '/jpg2'

    d = sio.loadmat(INPUT_MAT)
    r = d['gt'][0, 0]

    paths = sorted(glob(IMG_DIR + '/*.jpg'))

    assert len(paths) == r['set'].size

    train_data = []
    val_data = []
    test_data = []
    d = {
        1: train_data,
        2: val_data,
        3: test_data
    }
    for p, s, author_id in zip(paths, r['set'].ravel(), r['C'].ravel()):
        d[s].append({
                'class_id': author_id - 1,
                'path': p,
                'uid': p.split('/')[-1].split('.')[0]
            })

    def save_names(d, counts_dict, csv_name):
        classname_data = [{'class_id': author_id, 'class_name': classname, 'count': counts_dict[author_id]}
                          for author_id, classname in d.items()]
        pd.DataFrame(data=classname_data).to_csv(csv_name, index=False)
        pd.DataFrame(data=classname_data).to_csv(csv_name.replace('.csv', '.tsv'), index=False, sep='\t')

    names_dict = {author_id: classname[0] for author_id, classname in enumerate(r['Cnames'].ravel())}

    test_df = pd.DataFrame(data=test_data)
    train_df = pd.DataFrame(data=train_data)
    validation_df = pd.DataFrame(data=val_data)

    # Complete data (6.6k classes)
    test_df.to_csv('test.csv', index=False)
    train_df.to_csv('train.csv', index=False)
    validation_df.to_csv('validation.csv', index=False)
    d = defaultdict(lambda: 0)
    d.update(train_df.class_id.value_counts())
    save_names(names_dict, d, 'author_names.csv')

    def get_artist_ids_for_threshold_in_test(k=10):
        test_value_counts = test_df.class_id.value_counts()
        s = set(test_value_counts.index[test_value_counts>=k])
        artist_ids = s.intersection(train_df.class_id).intersection(validation_df.class_id)
        artist_ids.difference_update([6620, 6621]) # Remove unknown and anonime
        return artist_ids

    def get_top_k_artists(k):
        tmp = r['Ccount'][:-2]  # Remove 6620 and 6621 (unknwon and anonime)
        return set(np.argsort(tmp.ravel())[-k:])

    def get_artists_ids_for_threshold(k):
        tmp = r['Ccount'][:-2]  # Remove 6620 and 6621 (unknwon and anonime)
        return np.where(tmp >= k)[0]

    def reassign_df(df, artist_dict):
        new_df = df.copy()
        new_df = new_df[np.array([c in artist_dict.keys() for c in new_df.class_id], np.bool)]
        new_df.class_id = [artist_dict[c] for c in new_df.class_id]
        return new_df

    # 374
    new_artist_ids = get_artist_ids_for_threshold_in_test()
    d = {_id: i for i, _id in enumerate(new_artist_ids)}
    new_names_dict = {d[_id]: name for _id, name in names_dict.items() if _id in d.keys()}

    reassign_df(test_df, d).to_csv('test_374.csv', index=False)
    reassign_df(train_df, d).to_csv('train_374.csv', index=False)
    reassign_df(validation_df, d).to_csv('validation_374.csv', index=False)
    save_names(new_names_dict, reassign_df(train_df, d).class_id.value_counts(), 'author_names_374.csv')

    # 374+u
    unknown_id = len(new_artist_ids)
    for i in names_dict.keys():
        if i not in d.keys():
            d[i] = unknown_id
    new_names_dict[unknown_id] = 'Unknown'

    reassign_df(test_df, d).to_csv('test_374_u.csv', index=False)
    reassign_df(train_df, d).to_csv('train_374_u.csv', index=False)
    reassign_df(validation_df, d).to_csv('validation_374_u.csv', index=False)
    save_names(new_names_dict, reassign_df(train_df, d).class_id.value_counts(), 'author_names_374_u.csv')

    # 100, 200, 300
    for k in [100, 200, 300]:
        new_artist_ids = get_top_k_artists(k)
        d = {_id: i for i, _id in enumerate(new_artist_ids)}
        new_names_dict = {d[_id]: name for _id, name in names_dict.items() if _id in d.keys()}

        reassign_df(test_df, d).to_csv('test_{}.csv'.format(k), index=False)
        reassign_df(train_df, d).to_csv('train_{}.csv'.format(k), index=False)
        reassign_df(validation_df, d).to_csv('validation_{}.csv'.format(k), index=False)
        save_names(new_names_dict, reassign_df(train_df, d).class_id.value_counts(), 'author_names_{}.csv'.format(k))