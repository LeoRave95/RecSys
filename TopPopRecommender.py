import numpy as np

from Recommender import Recommender


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

