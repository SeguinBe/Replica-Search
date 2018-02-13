import os
from sacred import Experiment
from replica_learn.utils import TrainingParams, ModelParams

ex = Experiment()

PRETRAINED_DIR = '/mnt/cluster-nas/benoit/pretrained_nets'


@ex.config
def my_config():
    training_dir = '/scratch/benoit/authorship'
    author_csv = '/scratch/benoit/datasets/rijkschallenge/author_names.csv'
    training_csv = '/scratch/benoit/datasets/rijkschallenge/train.csv'
    validation_csv = '/scratch/benoit/datasets/rijkschallenge/validation.csv'
    testing_csv = '/scratch/benoit/datasets/rijkschallenge/test.csv'

    model_params = ModelParams().to_dict()  # Model parameters
    training_params = TrainingParams().to_dict()  # Training parameters


@ex.named_config
def embeddings():
    model_params = {
        'class_embedding_dim': 128,
        'fc_units': 1024
    }


@ex.named_config
def resnet_50():
    model_params = {
        'base_model': 'resnet50',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'resnet_v1_50.ckpt'),
        'pretrained_name_scope': 'resnet_v1_50',
        'learning_rate': 0.000005,
        'blocks': 4
    }


@ex.named_config
def rijks_374_u():
    author_csv = '/scratch/benoit/datasets/rijkschallenge/author_names_374_u.csv'
    training_csv = '/scratch/benoit/datasets/rijkschallenge/train_374_u.csv'
    validation_csv = '/scratch/benoit/datasets/rijkschallenge/validation_374_u.csv'
    testing_csv = '/scratch/benoit/datasets/rijkschallenge/test_374_u.csv'


@ex.named_config
def rijks_374():
    author_csv = '/scratch/benoit/datasets/rijkschallenge/author_names_374.csv'
    training_csv = '/scratch/benoit/datasets/rijkschallenge/train_374.csv'
    validation_csv = '/scratch/benoit/datasets/rijkschallenge/validation_374.csv'
    testing_csv = '/scratch/benoit/datasets/rijkschallenge/test_374.csv'


@ex.named_config
def rijks_100():
    author_csv = '/scratch/benoit/datasets/rijkschallenge/author_names_100.csv'
    training_csv = '/scratch/benoit/datasets/rijkschallenge/train_100.csv'
    validation_csv = '/scratch/benoit/datasets/rijkschallenge/validation_100.csv'
    testing_csv = '/scratch/benoit/datasets/rijkschallenge/test_100.csv'
    training_index_file = '/scratch/benoit/datasets/rijkschallenge/index_train_100.hdf5'
    validation_index_file = '/scratch/benoit/datasets/rijkschallenge/index_validation_100.hdf5'


@ex.named_config
def rijks_200():
    author_csv = '/scratch/benoit/datasets/rijkschallenge/author_names_200.csv'
    training_csv = '/scratch/benoit/datasets/rijkschallenge/train_200.csv'
    validation_csv = '/scratch/benoit/datasets/rijkschallenge/validation_200.csv'
    testing_csv = '/scratch/benoit/datasets/rijkschallenge/test_200.csv'


@ex.named_config
def rijks_300():
    author_csv = '/scratch/benoit/datasets/rijkschallenge/author_names_300.csv'
    training_csv = '/scratch/benoit/datasets/rijkschallenge/train_300.csv'
    validation_csv = '/scratch/benoit/datasets/rijkschallenge/validation_300.csv'
    testing_csv = '/scratch/benoit/datasets/rijkschallenge/test_300.csv'
