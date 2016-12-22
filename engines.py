import os
import pickle

import pandas as pd
import numpy as np

import xgboost as xgb
import lightgbm as lgbm

from utils import *


def xgboost(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    param = {
        'objective': 'multi:softprob',
        'eta': 0.1,
        'min_child_weight': 10,
        'max_depth': 8,
        'silent': 1,
        # 'nthread': 16,
        'eval_metric': 'mlogloss',
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
        'num_class': len(products),
    }

    if not restore:
        X_train = XY_train.as_matrix(columns=features)
        Y_train = XY_train.as_matrix(columns=["y"])
        W_train = XY_train.as_matrix(columns=["weight"])
        train = xgb.DMatrix(X_train, label=Y_train, feature_names=features, weight=W_train)

        X_validate = XY_validate.as_matrix(columns=features)
        Y_validate = XY_validate.as_matrix(columns=["y"])
        W_validate = XY_validate.as_matrix(columns=["weight"])
        validate = xgb.DMatrix(X_validate, label=Y_validate, feature_names=features, weight=W_validate)

        with Timer("train"):
            print(param)
            evallist  = [(train,'train'), (validate,'eval')]
            model = xgb.train(param, train, 1000, evals=evallist, early_stopping_rounds=20)
            pickle.dump(model, open("next_multi.pickle", "wb"))
    else:
        with Timer("restore model"):
            model = pickle.load(open("next_multi.pickle", "rb"))
    best_ntree_limit = model.best_ntree_limit

    if XY_all is not None:
        X_all = XY_all.as_matrix(columns=features)
        Y_all = XY_all.as_matrix(columns=["y"])
        W_all = XY_all.as_matrix(columns=["weight"])
        all_data = xgb.DMatrix(X_all, label=Y_all, feature_names=features, weight=W_all)

        evallist  = [(all_data,'all_data')]
        best_ntree_limit = int(best_ntree_limit * (len(XY_train) + len(XY_validate)) / len(XY_train))
        model = xgb.train(param, all_data, best_ntree_limit, evals=evallist)

    print("Feature importance:")
    for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):
        print(kv)

    X_test = test_df.as_matrix(columns=features)
    test = xgb.DMatrix(X_test, feature_names=features)

    return model.predict(test, ntree_limit=best_ntree_limit)


def lightgbm(XY_train, XY_validate, test_df, features, XY_all=None, restore=False):
    train = lgbm.Dataset(XY_train[list(features)], label=XY_train["y"], weight=XY_train["weight"], feature_name=features)
    validate = lgbm.Dataset(XY_validate[list(features)], label=XY_validate["y"], weight=XY_validate["weight"], feature_name=features, reference=train)

    params = {
        'task' : 'train',
        'boosting_type' : 'gbdt',
        'objective' : 'multiclass',
        'num_class': 24,
        'metric' : {'multi_logloss'},
        'is_training_metric': True,
        'max_bin': 255,
        'num_leaves' : 64,
        'learning_rate' : 0.1,
        'feature_fraction' : 0.8,
        'min_data_in_leaf': 10,
        'min_sum_hessian_in_leaf': 5,
        # 'num_threads': 16,
    }
    print(params)

    if not restore:
        with Timer("train lightgbm_lib"):
            model = lgbm.train(params, train, num_boost_round=1000, valid_sets=validate, early_stopping_rounds=20)
            best_iteration = model.best_iteration
            model.save_model("tmp/lgbm.model.txt")
            pickle.dump(best_iteration, open("tmp/lgbm.model.meta", "wb"))
    else:
        with Timer("restore lightgbm_lib model"):
            model = lgbm.Booster(model_file="tmp/lgbm.model.txt")
            best_iteration = pickle.load(open("tmp/lgbm.model.meta", "rb"))

    if XY_all is not None:
        best_iteration = int(best_iteration * len(XY_all) / len(XY_train))
        all_train = lgbm.Dataset(XY_all[list(features)], label=XY_all["y"], weight=XY_all["weight"], feature_name=features)
        with Timer("retrain lightgbm_lib with all data"):
            model = lgbm.train(params, all_train, num_boost_round=best_iteration)
        model.save_model("tmp/lgbm.all.model.txt")

    print("Feature importance by split:")
    for kv in sorted([(k,v) for k,v in zip(features, model.feature_importance("split"))], key=lambda kv: kv[1], reverse=True):
        print(kv)
    print("Feature importance by gain:")
    for kv in sorted([(k,v) for k,v in zip(features, model.feature_importance("gain"))], key=lambda kv: kv[1], reverse=True):
        print(kv)

    return model.predict(test_df[list(features)], num_iteration=best_iteration)
