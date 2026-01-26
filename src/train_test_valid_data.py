"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Splitting data to obtain training set, testing set, and validation set.
"""

################### Splitting data to obtain training set, testing set, and validation set ###################
##### When importing packages, avoid circular imports. #####
import random


def __random_index(data_len:str, test_radio=0.2, valid_radio=0.2, seed=42, showInfo=False):
    """
    Enter the length of a dataset and return the random index for dividing the training set, testing set, and validation set.
    
    :param data_len: The sample size or length of the dataset to be partitioned;
    :param test_radio: Test set percentage;
    :param valid_radio: Verification set percentage;
    :param seed: Random seed;
    :param showInfo: Do you want to output detailed information;
    :return: the random indices of the training set, testing set, and validation set.
    """
    index = list(range(data_len))
    # Randomly shuffle subscripts
    random.seed(seed)
    random.shuffle(index)
    # Default 20% of the data as test data
    test_num = int(data_len * test_radio)
    # Default 20% of data as validation set
    valid_num = int(data_len * valid_radio)
    test_index = index[0:test_num]
    valid_index = index[test_num:(test_num+valid_num)]
    train_index = index[(test_num+valid_num):]
    if showInfo:
        print("split data:")
        print("  Test set length:       {}({:.3f}%)".format(len(test_index), len(test_index)/data_len*100 ))
        print("  Validation set length: {}({:.3f}%)".format(len(valid_index), len(valid_index)/data_len*100 ))
        print("  Training set length:   {}({:.3f}%)".format(len(train_index), len(train_index)/data_len*100 ))
        print("  Total length:         ", len(test_index)+len(valid_index)+len(train_index))

    return test_index, valid_index, train_index


def get_train_test_valid_data(data, barcode, matrix, test_radio=0.2, valid_radio=0.2, seed=42, showInfo=False):
    """
    Obtain training set, testing set, and validation set.

    :param data: The dataset to be split (encoded);
    :param barcode: Barcode list of cells;
    :param matrix: Cell-SNP matrix;
    :param test_radio: Test set percentage;
    :param valid_radio: Verification set percentage;
    :param seed: Random seed;
    :param showInfo: Do you want to output detailed information;
    :return: the training set, testing set, and validation set.
    """
    test_index, valid_index, train_index = __random_index(len(data), test_radio, valid_radio, seed, showInfo)
    # data
    test_data = data[test_index]
    valid_data = data[valid_index]
    train_data = data[train_index]
    # barcode
    test_barcode = barcode[test_index]
    valid_barcode = barcode[valid_index]
    train_barcode = barcode[train_index]
    # matrix
    test_mtx = matrix[test_index]
    valid_mtx = matrix[valid_index]
    train_mtx = matrix[train_index]

    return (train_data,train_barcode,train_mtx),(valid_data,valid_barcode,valid_mtx),(test_data,test_barcode,test_mtx)

