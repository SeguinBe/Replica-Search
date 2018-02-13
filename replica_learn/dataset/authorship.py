from .base import *
from collections import defaultdict


class ClassificationDataset(Dataset):
    def __init__(self, data_dict: Union[Dict[str, Tuple[str, int]], str], classname_dict: Union[Dict[int, str], str]):
        if isinstance(data_dict, str):
            df = pd.read_csv(data_dict)
            raw_data = {r.uid: (r.path, r.class_id) for i, r in df.iterrows()}
        else:
            raw_data = data_dict
        path_dict = {uid: v[0] for uid, v in raw_data.items()}
        self.reverse_class_dict = {uid: v[1] for uid, v in raw_data.items()}
        self.class_dict = defaultdict(list)
        for uid, author in self.reverse_class_dict.items():
            self.class_dict[author].append(uid)

        if isinstance(classname_dict, str):
            df = pd.read_csv(classname_dict)
            self.classname_dict = {r.class_id: r.class_name for i, r in df.iterrows()}
        else:
            self.classname_dict = classname_dict

        super().__init__(path_dict=path_dict)

        assert self.get_number_classes() == max(*self.class_dict.keys())+1

    def get_number_classes(self):
        return len(self.classname_dict)

    def get_max_elements_in_class(self):
        return max([len(l) for l in self.class_dict.values()])

    def get_weight_vector(self):
        counts = np.ones(len(self.class_dict))
        for class_id, l in self.class_dict.items():
            counts[class_id] = len(l)
        weights = 1/counts
        weights *= np.mean(counts)
        return weights

    def generate_training_samples(self, id_only=False):
        return [(uid if id else self.path_dict[uid], class_id)
                for uid, class_id in self.reverse_class_dict.items()]

