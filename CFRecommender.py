import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from Recommender import Recommender


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
