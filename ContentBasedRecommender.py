import numpy as np
from tqdm import tqdm

from Recommender import Recommender


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
