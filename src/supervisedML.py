import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.compose import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
from sklearn.impute import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *


results = {
           "model" : [],
           "metric" : [],
           "test mean" : [],
           "train mean" : [],
           "fit time mean" : [],
           "score time mean" : [],
}
# if there are categorical features, consider OHE
# if there are missing values, consider imputation
# either ways, scaling is a good option
# @param dataframe of data, size to split train/test ~0.2, target to predict
# @return 
def startPredicttion(dataframe, size, target):
    train_df, test_df = train_test_split(dataframe, test_size=size)#, random_state=123)
    # print(train_df.info())
    y_train, y_test = train_df[target], test_df[target]
    x_train, x_test = train_df.drop(columns=[target]), test_df.drop(columns=[target])
    #
    cat_ft  = list(dataframe.select_dtypes('object'))
    num_ft  = list(filter(lambda x: x not in cat_ft, list(x_train)))
    # print(cat_ft, num_ft)
    lr_pipe = transformFeatures(cat_ft, num_ft, x_train, y_train)
    print("test score:", lr_pipe.score(x_test, y_test), "\n")
    print(
        classification_report(
            y_test, lr_pipe.predict(x_test), target_names=["Default=0", "Default=1"]
        )
    )


def transformFeatures(cat_ft, num_ft, data, target):
    preprocessor = make_column_transformer(
               (make_pipeline(SimpleImputer(strategy="constant"), StandardScaler()), num_ft),
               (make_pipeline(SimpleImputer(strategy="constant"), OneHotEncoder(sparse=False, handle_unknown="ignore")), cat_ft),
    )
    # transformed  = preprocessor.fit_transform(data)
    # hyperparameter tuning might be required
    lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=1000, random_state=123, C=0.1)) 
    #
    append_scores(results, "LogisticRegression", lr_pipe, data, target, return_train_score=True, cv = 5)
    #
    print(pd.DataFrame(results), "\n")
    lr_pipe.fit(data, target)

    return lr_pipe
    

def append_scores(dict_r, model_name, model, X_train, y_train, **kwargs):
    cv_scores = cross_validate(model, X_train, y_train, **kwargs, scoring = "accuracy")
    dict_r["model"].append(model_name)
    dict_r["metric"].append("accuracy")
    dict_r["train mean"].append      ( (f"%0.3f (+/- %0.3f)" % ((cv_scores)["train_score"].mean(), (cv_scores)["train_score"].std())) )
    dict_r["test mean"].append       ( (f"%0.3f (+/- %0.3f)" % ((cv_scores)["test_score"].mean(),  (cv_scores)["test_score"].std())) )
    dict_r["fit time mean"].append   ( (f"%0.3f (+/- %0.3f)" % ((cv_scores)["fit_time"].mean(),    (cv_scores)["fit_time"].std())) )
    dict_r["score time mean"].append ( (f"%0.3f (+/- %0.3f)" % ((cv_scores)["score_time"].mean(),  (cv_scores)["score_time"].std())) )
    return None


def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation
    """
    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)

#
#
main_df = pd.read_csv("../data/UCI_Credit_Card.csv")
startPredicttion(main_df, 0.2, "default.payment.next.month")