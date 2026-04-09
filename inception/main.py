from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES

from utils.utils import transform_labels
from utils.utils import create_directory
from utils.utils import run_length_xps
from utils.utils import generate_results_csv

import numpy as np
import os
import re
import sys
import sklearn


def get_next_results_dir(root_dir):
    pattern = re.compile(r'^results_improvement_(\d+)$')
    nums = [int(pattern.match(d).group(1)) for d in os.listdir(root_dir) if pattern.match(d)]
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(root_dir, f'results_improvement_{next_num}')


TARGET_DATASETS = ['GunPoint', 'ECG200', 'FordA']


def load_local_tsv_dataset(root_dir, dataset_name):
    dataset_dir = os.path.join(root_dir, 'data', dataset_name)

    train_path = os.path.join(dataset_dir, dataset_name + '_TRAIN.tsv')
    test_path = os.path.join(dataset_dir, dataset_name + '_TEST.tsv')

    x_train_raw = np.loadtxt(train_path, delimiter='\t')
    x_test_raw = np.loadtxt(test_path, delimiter='\t')

    y_train = x_train_raw[:, 0]
    x_train = x_train_raw[:, 1:]

    y_test = x_test_raw[:, 0]
    x_test = x_test_raw[:, 1:]

    return x_train, y_train, x_test, y_test


def prepare_data():
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    y_train, y_test = transform_labels(y_train, y_test)

    y_true = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)

    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    if len(x_train.shape) == 2:
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc


def fit_classifier():
    input_shape = x_train.shape[1:]

    classifier = create_classifier(
        classifier_name,
        input_shape,
        nb_classes,
        output_directory
    )

    classifier.fit(x_train, y_train, x_test, y_test, y_true)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True):
    if classifier_name == 'nne':
        from classifiers import nne
        return nne.Classifier_NNE(output_directory, input_shape,
                                  nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose=True, build=build)
    raise ValueError('Unknown classifier_name: {}'.format(classifier_name))


def get_xp_val(xp):
    if xp == 'batch_size':
        xp_arr = [16, 32, 128]
    elif xp == 'use_bottleneck':
        xp_arr = [False]
    elif xp == 'use_residual':
        xp_arr = [False]
    elif xp == 'nb_filters':
        xp_arr = [16, 64]
    elif xp == 'depth':
        xp_arr = [3, 9]
    elif xp == 'kernel_size':
        xp_arr = [8, 64]
    else:
        raise Exception('wrong argument')
    return xp_arr


############################################### main
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth',
       'kernel_size', 'batch_size']

if sys.argv[1] == 'InceptionTime':
    classifier_name = 'inception'
    archive_name = ARCHIVE_NAMES[0]
    nb_iter_ = 1

    datasets_dict = {}
    for ds_name in TARGET_DATASETS:
        datasets_dict[ds_name] = load_local_tsv_dataset(root_dir, ds_name)

    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)

        tmp_output_directory = get_next_results_dir(root_dir) + '/'
        
        for dataset_name in TARGET_DATASETS:
            print('\n==============================')
            print(f"Starting training for {dataset_name}")
            print('==============================')

            x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

            output_directory = tmp_output_directory + dataset_name + '/'

            temp_output_directory = create_directory(output_directory)

            if temp_output_directory is None:
                print('Already_done', tmp_output_directory, dataset_name)
                continue

            fit_classifier()

            print(f"Finished training for {dataset_name}")
            print('\t\t\t\tDONE')


elif sys.argv[1] == 'InceptionTime_xp':
    classifier_name = 'inception'
    archive_name = 'TSC'
    max_iterations = 1

    datasets_dict = {}
    for ds_name in TARGET_DATASETS:
        datasets_dict[ds_name] = load_local_tsv_dataset(root_dir, ds_name)

    for xp in xps:
        xp_arr = get_xp_val(xp)

        print('xp', xp)

        for xp_val in xp_arr:
            print('\txp_val', xp_val)

            kwargs = {xp: xp_val}

            for iter in range(max_iterations):
                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)
                print('\t\titer', iter)

                for dataset_name in TARGET_DATASETS:
                    output_directory = (
                        root_dir + '/results/' + classifier_name + '/' + xp + '/' +
                        str(xp_val) + '/' + archive_name + trr + '/' + dataset_name + '/'
                    )

                    print('\t\t\tdataset_name', dataset_name)
                    x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

                    temp_output_directory = create_directory(output_directory)

                    if temp_output_directory is None:
                        print('\t\t\t\t', 'Already_done')
                        continue

                    input_shape = x_train.shape[1:]

                    from classifiers import inception

                    classifier = inception.Classifier_INCEPTION(
                        output_directory,
                        input_shape,
                        nb_classes,
                        verbose=False,
                        build=True,
                        **kwargs
                    )

                    classifier.fit(x_train, y_train, x_test, y_test, y_true)

                    create_directory(output_directory + '/DONE')

                    print('\t\t\t\t', 'DONE')

elif sys.argv[1] == 'run_length_xps':
    run_length_xps(root_dir)

elif sys.argv[1] == 'generate_results_csv':
    clfs = []
    itr = '-0-'
    inception_time = 'inception'

    clfs.append(inception_time + itr)

    for xp in xps:
        xp_arr = get_xp_val(xp)
        for xp_val in xp_arr:
            clfs.append(inception_time + '/' + xp + '/' + str(xp_val) + itr)

    df = generate_results_csv('results.csv', root_dir, clfs)
    print(df)

else:
    raise ValueError('Unknown command: {}'.format(sys.argv[1]))