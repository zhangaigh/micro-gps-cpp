import pickle
import numpy as np

def save_obj(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def compute_precision_recall(true_flags, test_scores, steps=1000):
    
    thresh_array = np.linspace(np.amin(test_scores), np.amax(test_scores), steps)
    # print(thresh_array)
    precision = np.zeros(thresh_array.size, dtype=np.float32)
    recall = np.zeros(thresh_array.size, dtype=np.float32)

    true_flags = true_flags > 0
    for thresh_idx in range(1, len(thresh_array)):
        test_flags = test_scores < thresh_array[thresh_idx]

        # print(test_flags)
        # print(np.logical_and(true_flags, test_flags).ravel().nonzero())
        num_true_positives = float(np.logical_and(true_flags, test_flags).nonzero()[0].size)
        num_test_positives = float(test_flags.nonzero()[0].size)
        num_all_positives = float(true_flags.nonzero()[0].size)

        print('true positives = {}, test positives = {}, all positives = {}'.format(num_true_positives, num_test_positives, num_all_positives))

        if num_test_positives == 0:
            num_test_positives = 1
        if num_all_positives == 0:
            num_all_positives= 1


        precision[thresh_idx] = num_true_positives / num_test_positives
        recall[thresh_idx] = num_true_positives / num_all_positives

        if num_true_positives == num_all_positives:
            break

    
    precision = precision[recall > 0]
    recall = recall[recall > 0]

    return precision, recall
