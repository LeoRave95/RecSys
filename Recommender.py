import datetime
import tqdm
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import time

from Compute_Similarity import Compute_Similarity


def precision(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def MAP(recommended_items, relevant_items):
    is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


class Recommender(object):

    def __init__(self, URM, ICM):
        self.URM = URM
        self.ICM = ICM

    def load_model(self, URM_path='snapshots/URM.npz', R_path='snapshots/R.npz', verbose=True):

        if verbose:
            print('Loading URM...')

        self.URM = sparse.load_npz(URM_path)

        if verbose:
            print('Loading R...')

        self.R = sparse.load_npz(R_path)

        if verbose:
            print('Model loaded.')

    def save_model(self, verbose=True):
        if verbose:
            print('Saving model to file...')

        t = time.time()
        timestamp = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d_%H:%M:%S')

        URM_path = 'snapshots/URM_%s' % timestamp
        R_path = 'snapshots/R_%s' % timestamp

        sparse.save_npz(URM_path, self.URM)
        sparse.save_npz(R_path, self.R)

    """Fit the model"""

    def fit(self):
        raise NotImplementedError("Should have implemented this")

    """Fetch n recommended items"""

    def recommend(self, user_id, at):
        raise NotImplementedError("Should have implemented this")

    '''testing performances of the recommender'''

    def evaluate(self, URM_test, user_list_unique, at=10, verbose=True):
        cumulative_precision = 0.0
        cumulative_recall = 0.0
        cumulative_MAP = 0.0

        user_list_len = len(user_list_unique)
        num_eval = 0

        iterable = user_list_unique
        if verbose:
            iterable = tqdm.tqdm(iterable)

        for user_id in iterable:

            relevant_items = URM_test[user_id].indices

            if len(relevant_items) > 0:
                recommended_items = self.recommend(user_id, at=at)
                num_eval += 1

                cumulative_precision += precision(recommended_items, relevant_items)
                cumulative_recall += recall(recommended_items, relevant_items)
                cumulative_MAP += MAP(recommended_items, relevant_items)

        cumulative_precision /= num_eval
        cumulative_recall /= num_eval
        cumulative_MAP /= num_eval

        if verbose:
            print("Recommender performance is: Precision = {:.8f}, Recall = {:.8f}, MAP = {:.8f}".format(
                cumulative_precision, cumulative_recall, cumulative_MAP))

        return cumulative_precision, cumulative_recall, cumulative_MAP
