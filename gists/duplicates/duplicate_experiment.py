import pandas as pd
import shelve
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm, ensemble
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn
import pickle

COMPUTE = True

if COMPUTE:
    INPUT_CSV = '/mnt/cluster-nas/benoit/duplicate_exp.csv'
    INPUT_FEATURES = '/scratch/benoit/duplicate_features_exp.shl'

    # Load csv
    # Columns img_uid1, img_uid2, key, is_duplicate, is_training
    data = pd.read_csv(INPUT_CSV)
    labels = data.is_duplicate

    # Load features (dict of (features, boxes))
    with shelve.open(INPUT_FEATURES, flag='r') as f:
        features = np.array([
            f[k][0] for k in data.key
        ])

    print('Proportion of duplicates: {}'.format(np.mean(data.is_duplicate)))
    print('{} training, {} testing'.format(np.sum(data.is_training), np.sum(~data.is_training)))

    # Possible classifiers
    clf_dict = dict()
    for kernel_type in ['rbf']:#['linear', 'rbf']:
        for C in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
            clf_dict[(kernel_type, C)] = svm.SVC(kernel=kernel_type, C=C, probability=True)
    clf_dict[('GB',)] = ensemble.GradientBoostingClassifier()

    # Not matching of small overlap
    data['is_detail'] = ((features[:, 3] < 0.3) | (features[:, 2] <= 10)) & data.is_duplicate
    data['duplicate_no_matches'] = (features[:, 2] <= 10) & data.is_duplicate
    print('Proportion of details: {}'.format(np.sum(data.is_detail)/np.sum(data.is_duplicate)))

    # Scaling features
    class MyPreprocess:
        def transform(self, features):
            features = features.copy()
            features[:, :3] = np.log(features[:, :3])
            features[:, 3:5] = np.sqrt(features[:, 3:5])
            return features

        def fit(self, X, y=None):
            pass


    features = MyPreprocess().transform(features)
    original_features = features.copy()

    # all_features = np.concatenate([all_features, all_features**2], axis=1)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)


    def get_best_classifier(training_mask):
        training_features = features[training_mask]
        training_labels = labels[training_mask]
        scores_dict = dict()
        print("Training on {} elements, including {} positive".format(len(training_labels), np.sum(training_labels)))
        for k, clf in tqdm(clf_dict.items()):
            scores = cross_val_score(clf, training_features, training_labels, cv=10, scoring='f1')
            #print("{} | F-score: {:.4f} (+/- {:.2f})".format(k, scores.mean(), scores.std() * 2))
            scores_dict[k] = scores.mean()
        best_key = max(scores_dict.keys(), key=scores_dict.get)
        print(best_key)
        return clf_dict[best_key].fit(training_features, training_labels)


    def get_scores(best_clf, testing_mask):
        testing_features = features[testing_mask]
        testing_labels = labels[testing_mask]
        testing_details = data.is_detail[testing_mask]
        scores = best_clf.predict_proba(testing_features)[:, 1]
        precisions, recalls, thresholds = precision_recall_curve(testing_labels, scores)
        # print(precisions, recalls, thresholds)
        for precision_threshold in (0.95, 0.98, 0.99, 0.995, 0.999, 1.0):
            precision_threshold_idx = np.min(np.where(precisions >= precision_threshold)[0])
            threshold_value = thresholds[precision_threshold_idx]
            actual_recall = recalls[precision_threshold_idx]
            recall_detail = np.mean(scores[testing_details] >= threshold_value)
            recall_non_detail = np.mean(scores[testing_labels & ~testing_details] >= threshold_value)
            print('P={:.3f} | t={:.4f} | R_total={:.3f} | R_not_detail={:.3f} | R_detail={:.3f} | #FN={} | #FP={}'.format(
                precision_threshold, threshold_value, actual_recall, recall_non_detail, recall_detail,
                (1 - actual_recall) * np.sum(testing_labels), np.sum((scores >= threshold_value) & ~testing_labels))
            )
        return precisions, recalls, thresholds


    curves = dict()

    # With details
    best_clf = get_best_classifier(data.is_training)
    curves['spatial_histo'] = get_scores(best_clf, ~data.is_training)

    clf_pipeline = Pipeline([
        ('preprocess', MyPreprocess()),
        ('scaler', scaler),
        ('clf', best_clf)])
    with open('duplicate_classifier.pkl', 'wb') as f:
        pickle.dump(clf_pipeline, f)


    # Without details
    # best_clf = get_best_classifier(data.is_training & ~data.duplicate_no_matches)
    # get_scores(best_clf, ~data.is_training)

    # No thresholds
    features = original_features[:, [0, 1, 2, 3, 4, 6]]
    best_clf = get_best_classifier(data.is_training)
    curves['spatial'] = get_scores(best_clf, ~data.is_training)

    # No spatial spread
    features = original_features[:, [0, 1, 2] + list(range(5, original_features.shape[1]))]
    best_clf = get_best_classifier(data.is_training)
    curves['histo'] = get_scores(best_clf, ~data.is_training)

    # None
    features = original_features[:, [0, 1, 2, 6]]
    best_clf = get_best_classifier(data.is_training)
    curves['base'] = get_scores(best_clf, ~data.is_training)

    with open('duplicate_experiment_stats.pkl', 'wb') as f:
        pickle.dump(curves, f)

with open('duplicate_experiment_stats.pkl', 'rb') as f:
    curves = pickle.load(f)


keys = ['base', 'spatial', 'histo', 'spatial_histo']
labels = ['Single Threshold', '+ Spatial Spread', 'Multiple Thresholds', '+ Spatial Spread']
for k, color in zip(keys, ['green', 'blue', 'red', 'black']):
    plt.plot(curves[k][1], curves[k][0], color=color)
    plt.xlim([0.9, 1.0])
    plt.xlabel('Recall')
    plt.ylim([0.95, 1.0])
    plt.ylabel('Precision')

plt.legend(labels)
plt.savefig('duplicate_experiment_curves.pdf')
