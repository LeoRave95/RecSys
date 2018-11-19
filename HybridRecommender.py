import numpy as np
from tqdm import tqdm

from Compute_Similarity import Compute_Similarity
from Recommender import Recommender


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
        similarity_object_album = Compute_Similarity(self.ICM.T, shrink=shrink, topK=self.ICM.shape[0],
                                                     normalize=normalize, similarity=similarity)
        S_ICM_album = similarity_object_album.compute_similarity()
        # similarity_object_artist = Compute_Similarity(ICM_artist.T, shrink=shrink, topK=ICM_artist.shape[0], normalize=normalize, similarity=similarity)
        # S_ICM_artist = similarity_object_artist.compute_similarity()
        # similarity_object_duration = Compute_Similarity(ICM_duration.T, shrink=shrink, topK=topK, normalize=normalize, similarity=similarity)
        # S_ICM_duration = similarity_object_duration.compute_similarity()

        # S_ICM_album = S_ICM_album.multiply(.8)
        # S_ICM_artist = S_ICM_artist.multiply(.2)
        # S_ICM_duration = S_ICM_duration.multiply(.0)

        S_ICM = S_ICM_album # + S_ICM_artist

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
