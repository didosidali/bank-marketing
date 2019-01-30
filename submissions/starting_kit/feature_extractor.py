from scipy import constants


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y):
        return self

    def transform(self, X_df):
        X_df_new = X_df.copy()
        X_df_new.job.replace(('admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
        X_df_new.marital.replace(('divorced','married','single','unknown'),(1,2,3,4), inplace=True)
        X_df_new.education.replace(('basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown'),(1,2,3,4,5,6,7,8), inplace=True)
        X_df_new.default.replace(('no','yes','unknown'),(1,2,3), inplace=True)
        X_df_new.housing.replace(('no','yes','unknown'),(1,2,3), inplace=True)
        X_df_new.loan.replace(('no','yes','unknown'),(1,2,3), inplace=True)
        X_df_new.contact.replace(('cellular','telephone'),(1,2), inplace=True)
        X_df_new.month.replace(('jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'),(1,2,3,4,5,6,7,8,9,10,11,12), inplace=True)
        X_df_new.day_of_week.replace(('mon','tue','wed','thu','fri'),(1,2,3,4,5), inplace=True)
        X_df_new.poutcome.replace(('failure','nonexistent','success'),(1,2,3), inplace=True)
        return X_df_new


