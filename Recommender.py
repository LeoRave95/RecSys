import datetime
import tqdm
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import time

from Compute_Similarity import Compute_Similarit


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


'''Random Recommender'''


class RandomRecommender(Recommender):

    def fit(self):
        self.num_items = self.URM.shape[0]

    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.num_items, at)

        return recommended_items


'''TopPopRecommender'''


class TopPopRecommender(Recommender):

    def fit(self):

        item_popularity = (self.URM > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()

        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, at=10, remove_seen=True):

        if remove_seen:
            unseen_items_mask = np.in1d(
                self.popular_items, self.URM_train[user_id].indices, assume_unique=True, invert=True)
            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popular_items[0:at]

        return recommended_items


'''ContentBasedRecommender'''


class ContentBasedRecommender(Recommender):

    def __knn(self, M, k=10, verbose=True):

        KNN = np.zeros([M.shape[0], k])

        rows, _ = M.nonzero()

        iterable = set(rows)
        if verbose:
            iterable = tqdm.tqdm(iterable)

        for r in iterable:
            row = M.getrow(r)
            row_dense = row.todense()
            row_dense = row_dense.argsort()
            row_dense = np.flip(row_dense, axis=0)
            row_dense = row_dense[0, :k]
            KNN[r] = row_dense

        return M

    def fit(self, k=10, verbose=True):

        if verbose:
            print('Starting model fitting...')
        '''
        S_ICM = sparse.coo_matrix((self.ICM[0][0].shape[0], self.ICM[0][0].shape[0]))
        S_ICM = S_ICM.tocsr()
        for M, w in ICM:
            cs = cosine_similarity(M, dense_output=False)
            S_ICM += cs.multiply(w)

        if verbose:
            print('Similarity matrix computed...')

        # TODO: improve taking into account KNN
        # S = self.__knn(S)

        if verbose:
            print('KNN computed...')

        R_ICM = np.dot(self.URM, S_ICM)

        if verbose:
            print('Expected rating matrix R_ICM computed...')

        self.R = R_ICM'''

        if verbose:
            print('Done!')

    def recommend(self, user_id, at=10, remove_seen=True, verbose=False):

        if verbose:
            print('Getting recommendations for user %i' % user_id)

        row = self.R[user_id]
        row_dense = np.array(row).squeeze()
        row_dense = row.todense()
        row_dense = np.argsort(row_dense)
        row_dense = np.flip(row_dense)

        if remove_seen:
            items = np.array(row_dense).squeeze()
            unseen_items_mask = np.in1d(items, self.URM[user_id].indices, assume_unique=True, invert=True)
            unseen_items = items[unseen_items_mask]
            recommended_items = unseen_items[:at]
        else:
            recommended_items = self.R[user_id, :at].todense()
            recommended_items = np.array(recommended_items).squeeze()

        return list(map(int, recommended_items))


'''Collaborative Filtering (Item based or User based)'''


class CFRecommender(Recommender):

    def __knn(self, M, k=200, verbose=True):

        KNN = np.zeros([M.shape[0], k])

        rows, _ = M.nonzero()

        iterable = set(rows)
        if verbose:
            iterable = tqdm.tqdm(iterable)

        for r in iterable:
            row = M.getrow(r)
            row_dense = row.todense()
            row_dense = row_dense.argsort()
            row_dense = np.flip(row_dense, axis=0)
            row_dense = row_dense[0, :k]
            KNN[r] = row_dense

        return M

    def fit(self, user_list_unique, ItemBased=True, verbose=True):

        if verbose:
            print('Starting model fitting...')

        if ItemBased:
            S_Item = cosine_similarity(self.URM.T, dense_output=False)
            # S_Item = self.__knn(S_Item)
        else:
            S_User = cosine_similarity(self.URM, dense_output=False)
            # S_User = self.__knn(S_User)

        if verbose:
            print('Similarity matrix computed...')

        # TODO: improve taking into account KNN

        if verbose:
            print('KNN computed...')

        if ItemBased:
            R_Item = np.dot(self.URM, S_Item)
        else:
            R_User = np.dot(S_User, self.URM)

        if verbose:
            print('Expected rating matrix R computed...')

        self.R = R_Item

        if verbose:
            print('Done!')

    def recommend(self, user_id, at=10, remove_seen=True, verbose=False):

        if verbose:
            print('Getting recommendations for user %i' % user_id)

        row = self.R[user_id]
        row_dense = np.array(row).squeeze()
        row_dense = row.todense()
        row_dense = np.argsort(row_dense)
        row_dense = np.flip(row_dense)

        if remove_seen:
            items = np.array(row_dense).squeeze()
            unseen_items_mask = np.in1d(items, self.URM[user_id].indices, assume_unique=True, invert=True)
            unseen_items = items[unseen_items_mask]
            recommended_items = unseen_items[:at]
        else:
            recommended_items = self.R[user_id, :at].todense()
            recommended_items = np.array(recommended_items).squeeze()

        return list(map(int, recommended_items))


'''Hybrid Recommender: Content Based + Item Based Collaborative Filtering'''


class HybridRecommender(Recommender):

    def __knn(self, M, k=15, verbose=True):

        KNN = np.zeros([M.shape[0], k])

        rows, _ = M.nonzero()

        iterable = set(rows)
        if verbose:
            iterable = tqdm.tqdm(iterable)

        for r in iterable:
            row = M.getrow(r)
            row_dense = row.todense()
            row_dense = row_dense.argsort()
            row_dense = np.flip(row_dense, axis=0)
            row_dense = row_dense[0, :k]
            KNN[r] = row_dense

        return KNN

    def fit(self, topK=50, shrink=100, normalize=True, similarity='cosine', verbose=True, alpha=0.95):

        # self.URM = URM_train

        # similarity_object_User = Compute_Similarity(self.URM.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)
        # S_User = similarity_object_User.compute_similarity()

        # R_User = np.dot(S_User, self.URM)

        # CF Item base
        # similarity_object = Compute_Similarity(self.URM, shrink=shrink, topK=self.URM.shape[1], normalize=normalize, similarity=similarity)
        # S_Item_CF = similarity_object.compute_similarity()
        # S_Item_CF = cosine_similarity(self.URM.transpose(), dense_output=False)

        # if verbose:
        #    print('CF Similarity done', S_Item_CF.shape)

        # Content base

        '''
        S_ICM = sparse.coo_matrix((self.ICM[0][0].shape[0], ICM[0][0].shape[0]))
        S_ICM = S_ICM.tocsr()
        for M, w in self.ICM:
            print(M, w)
            # similarity = cosine_similarity(M, dense_output=False)
            similarity_object = Compute_Similarity(M.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)
            similarity = similarity_object.compute_similarity()
            S_ICM += similarity.multiply(w)
        '''
        similarity_object_album = Compute_Similarit(self.ICM.T, shrink=shrink, topK=self.ICM.shape[0],
                                                     normalize=normalize, similarity=similarity)
        S_ICM_album = similarity_object_album.compute_similarity()
        # similarity_object_artist = Compute_Similarity(ICM_artist.T, shrink=shrink, topK=ICM_artist.shape[0], normalize=normalize, similarity=similarity)
        # S_ICM_artist = similarity_object_artist.compute_similarity()
        # similarity_object_duration = Compute_Similarity(ICM_duration.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)
        # S_ICM_duration = similarity_object_duration.compute_similarity()

        # S_ICM_album = S_ICM_album.multiply(.8)
        # S_ICM_artist = S_ICM_artist.multiply(.2)
        # S_ICM_duration = S_ICM_duration.multiply(.0)

        S_ICM = S_ICM_album  # + S_ICM_artist

        # similarity_object_User = Compute_Similarity(self.URM, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)
        # S_Item_CF = similarity_object.compute_similarity()

        if verbose:
            print('ICM done', S_ICM.shape)
            # print('S_Item_CF:', S_Item_CF)
            # print('S_ICM:', S_ICM)

        # S_Item_CF = S_Item_CF.multiply(alpha)
        # S_ICM = S_ICM.multiply(1-alpha)
        # S_Hybrid = S_Item_CF + S_ICM

        '''
        row, col = S_Hybrid.nonzero()

        for r,c in tqdm.tqdm(zip(row, col)):
            S_Hybrid[r,c] = 0.8*S_Hybrid[r,c] + 0.2*item_popularity[c]
        '''

        if verbose:
            print('Hybrid done')

        # R_Item = np.dot(self.URM, S_Hybrid)
        R_Item = np.dot(self.URM, S_ICM)

        if verbose:
            print('Expected rating matrix R computed...')

        # TODO: tune the parameter for User and Item mix
        # R_Item = R_Item.multiply(.7)
        # R_User = R_User.multiply(.3)

        # self.R = R_Item + R_User
        self.R = R_Item

    # TODO: implement hybrid recommendation
    def recommend(self, user_id, at=10, remove_seen=True, verbose=False):

        if verbose:
            print('Getting recommendations for user %i' % user_id)

        row = self.R[user_id]
        row_dense = np.array(row).squeeze()
        row_dense = row.todense()
        # row_dense = np.multiply(row_dense, alpha/np.amax(row_dense)) + np.multiply(self.item_popularity, (1-alpha)/np.amax(self.item_popularity))
        row_dense = np.argsort(row_dense)
        row_dense = np.flip(row_dense, axis=1)

        if remove_seen:
            items = np.array(row_dense).squeeze()
            unseen_items_mask = np.in1d(items, self.URM[user_id].indices, assume_unique=True, invert=True)
            unseen_items = items[unseen_items_mask]
            recommended_items = unseen_items[:at]
        else:
            recommended_items = self.R[user_id, :at].todense()
            recommended_items = np.array(recommended_items).squeeze()

        return list(map(int, recommended_items))
