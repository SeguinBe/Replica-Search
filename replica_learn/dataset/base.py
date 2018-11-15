from typing import Dict, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import json
from PIL import Image
import cv2

try:
    from pylatex import Document, Section, Tabular, StandAloneGraphic, NoEscape, NewLine, HFill, Package
except:
    pass
import networkx as nx
from tqdm import tqdm
import pandas as pd
from scipy.misc import imread, imresize, imsave
import os


class Dataset:
    def __init__(self, path_dict: Union[Dict[str, str], str]):
        if isinstance(path_dict, str):
            df = pd.read_csv(path_dict)
            self.path_dict = {r.uid: r.path for i, r in df.iterrows()}
        else:
            self.path_dict = path_dict

    def __repr__(self):
        return 'Dataset {} elements total'.format(len(self.path_dict))

    def get_img(self, uuid: str, max_dim=None):
        img = imread(self.get_image_path(uuid))

        def resize(img, max_dim=720):
            if img.shape[0] < img.shape[1]:
                new_size = (max_dim, int(img.shape[0]/img.shape[1]*max_dim))
            else:
                new_size = (int(img.shape[1]/img.shape[0]*max_dim), max_dim)
            return cv2.resize(img, new_size)

        if max_dim is not None:
            img = resize(img, max_dim)
        return img

    def plot_pair(self, uuid1, uuid2, label=None):
        plt.subplot(1, 2, 1)
        self.plot_img(uuid1)
        plt.subplot(1, 2, 2)
        self.plot_img(uuid2)
        if label is not None:
            plt.title('Label : {}'.format(label))

    def plot_img(self, uuid: str, box=None):
        arr = self.get_img(uuid)
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

    def pdf_export_groups(self, groups, filename, tmp_dir='/tmp/replica_tmp', split_files_into: Optional[int]=None,
                          labels: Optional[Dict[str, str]]=None, links: Optional[Dict[str, str]]=None):
        def _generate_one_file(groups, filename):
            geometry_options = {"rmargin": "1cm", "lmargin": "1cm"}
            doc = Document(geometry_options=geometry_options)
            doc.packages.append(Package('hyperref'))

            os.makedirs(tmp_dir, exist_ok=True)
            desired_size = 600 * 400

            def get_resized_img_path(uid):
                p = os.path.join(tmp_dir, '{}.jpg'.format(uid))
                if os.path.exists(p):
                    return p
                img = self.get_img(uid)
                s = np.array(img.shape[:2])
                s = np.round(s * np.sqrt(desired_size / np.prod(s))).astype(np.int32)
                img = imresize(img, s)
                imsave(p, img)
                return p

            def draw_set(uid_set):
                #if len(uid_set) == 2 and False:
                #    l = core_server.model.VisualLink.get_from_images(*list(uid_set))
                #    if l is not None:
                #        doc.append('Already present : {}'.format(l.type))
                with doc.create(Tabular('|' + 'p{4.5cm}|' * len(uid_set))) as table:
                    table.add_hline()
                    # print(StandAloneGraphic(path_dict[uid],
                    # image_options=NoEscape(r'width=0.12\textwidth')) for uid in uid_set))
                    table.add_row([StandAloneGraphic(get_resized_img_path(uid),
                                                     image_options=NoEscape(r'width=4cm')) for uid in uid_set])
                    table.add_hline()
                    if labels is not None:
                        table.add_row([labels[uid] for uid in uid_set])
                        table.add_hline()
                    if links is not None:
                        filter_if_no_link = lambda s: NoEscape(r'\href{{{}}}{{link}}'.format(s)) if s else ""
                        table.add_row([filter_if_no_link(links[uid]) for uid in uid_set])
                        table.add_hline()

            n_elements = 0
            max_per_line = 4
            for uid_set in tqdm(groups):
                # if len(uid_set)+n_elements>max_per_line:
                #    n_elements = 0
                #    doc.append(NewLine())
                # else:
                #    doc.append(HFill())
                for i in range(0, len(uid_set), max_per_line):
                    draw_set(list(uid_set)[i:i + max_per_line])
                    doc.append(NewLine())
                doc.append(NewLine())
                n_elements += len(uid_set)
            doc.generate_pdf(filename)

        if split_files_into is not None:
            for i in tqdm(range(0, len(groups), split_files_into)):
                _generate_one_file(groups[i:i + split_files_into],
                              '{}_{:04d}-{:04d}'.format(filename, i, min(i + split_files_into - 1, len(groups))))
        else:
            _generate_one_file(groups, filename)

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