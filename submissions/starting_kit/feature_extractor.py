from scipy import constants


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        # X_df_new = compute_rolling_std(X_df_new, 'Beta', '2h')
        return X_df_new

