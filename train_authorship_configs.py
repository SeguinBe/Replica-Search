import os
from sacred import Experiment

ex = Experiment()

PRETRAINED_DIR = '/mnt/cluster-nas/benoit/pretrained_nets'


@ex.config
def my_config():
    training_dir = '/scratch/benoit/authorship'
    author_csv = '/scratch/benoit/datasets/rijkschallenge/author_names.csv'
    training_csv = '/scratch/benoit/datasets/rijkschallenge/train.csv'
    validation_csv = '/scratch/benoit/datasets/rijkschallenge/validation.csv'
    testing_csv = '/scratch/benoit/datasets/rijkschallenge/test.csv'
    image_resizing = {
        'max_size': 320,
        'increment': 32
    }
    data_augmentation = {
        'activated': True,
        'include_flip_lr': True,
        'include_rotation': False,  # Might cause issue for narrow images (create 0 sized images)
        'include_zoom': False,
        'include_color': False
    }
    model_params = {
        'base_model': 'vgg16',
        'pretrained_file': os.path.join(PRETRAINED_DIR, 'vgg_16.ckpt'),
        'pretrained_name_scope': 'vgg_16',
        'reducing_op': 'avg',
        'blocks': 5,
        'batch_size': 64,
        'learning_rate': 0.000005,
        'decay_rate': 0.95,
        'decay_steps': 'auto',
        'train_batch_norm': False,
        'class_embedding_dim': 0,
        'weighted_loss': True
    }
    nb_epochs = 30


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
