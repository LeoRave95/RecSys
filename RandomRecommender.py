from Recommender import Recommender

'''Random Recommender'''


class RandomRecommender(Recommender):

    def fit(self):
        self.num_items = self.URM.shape[0]

    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.num_items, at)

        return recommended_items
