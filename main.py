
import gzip
import math
import pickle
import zlib
import io

import pandas as pd
import numpy as np

# import scipy.stats

from sklearn.preprocessing import LabelEncoder

import engines
from utils import *

np.random.seed(2016)

transformers = {}

def assert_uniq(series, name):
    uniq = np.unique(series, return_counts=True)
    print("assert_uniq", name, uniq)

def custom_one_hot(df, features, name, names, dtype=np.int8, check=False):
    for n, val in names.items():
        new_name = "%s_%s" % (name, n)
        print(name, new_name)
        df[new_name] = df[name].map(lambda x: 1 if x == val else 0).astype(dtype)

        if check:
            assert_uniq(df[new_name], new_name)
        features.append(new_name)


def label_encode(df, features, name):
    df[name] = df[name].astype('str')
    if name in transformers: # test
        df[name] = transformers[name].transform(df[name])
    else: # train
        transformers[name] = LabelEncoder()
        df[name] = transformers[name].fit_transform(df[name])
    features.append(name)

def encode_top(s, count=100, dtype=np.int8):
    uniqs, freqs = np.unique(s, return_counts=True)
    top = sorted(zip(uniqs,freqs), key=lambda vk: vk[1], reverse = True)[:count]
    top_map = {uf[0]: l+1 for uf, l in zip(top, range(len(top)))}
    return s.map(lambda x: top_map.get(x, 0)).astype(dtype)

def apply_transforms(train_df):
    features = []
    with Timer("apply transforms"):
        label_encode(train_df, features, "canal_entrada")
        # label_encode(train_df, features, "nomprov") # use cod_prov only
        label_encode(train_df, features, "pais_residencia")

        train_df["age"] = train_df["age"].fillna(0.0).astype(np.int16)
        features.append("age")

        train_df["renta"].fillna(1.0, inplace=True)
        train_df["renta_top"] = encode_top(train_df["renta"])
        assert_uniq(train_df["renta_top"], "renta_top")
        features.append("renta_top")
        train_df["renta"] = train_df["renta"].map(math.log)
        features.append("renta")

        train_df["antiguedad"] = train_df["antiguedad"].map(lambda x: 0.0 if x < 0 or math.isnan(x) else x+1.0).astype(np.int16)
        features.append("antiguedad")

        train_df["tipodom"] = train_df["tipodom"].fillna(0.0).astype(np.int8)
        features.append("tipodom")

        train_df["cod_prov"] = train_df["cod_prov"].fillna(0.0).astype(np.int8)
        features.append("cod_prov")

        train_df["fecha_dato_month"] = train_df["fecha_dato"].map(lambda x: int(x.split("-")[1])).astype(np.int8)
        features.append("fecha_dato_month")
        train_df["fecha_dato_year"] = train_df["fecha_dato"].map(lambda x: float(x.split("-")[0])).astype(np.int16)
        features.append("fecha_dato_year")
        train_df["fecha_alta_month"] = train_df["fecha_alta"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[1])).astype(np.int8)
        features.append("fecha_alta_month")
        train_df["fecha_alta_year"] = train_df["fecha_alta"].map(lambda x: 0.0 if x.__class__ is float else float(x.split("-")[0])).astype(np.int16)
        features.append("fecha_alta_year")

        train_df["fecha_dato_float"] = train_df["fecha_dato"].map(date_to_float)
        train_df["fecha_alta_float"] = train_df["fecha_alta"].map(date_to_float)

        train_df["dato_minus_alta"] = train_df["fecha_dato_float"] - train_df["fecha_alta_float"]
        features.append("dato_minus_alta")

        train_df["int_date"] = train_df["fecha_dato"].map(date_to_int).astype(np.int8)

        custom_one_hot(train_df, features, "indresi", {"n":"N"})
        custom_one_hot(train_df, features, "indext", {"s":"S"})
        custom_one_hot(train_df, features, "conyuemp", {"n":"N"})
        custom_one_hot(train_df, features, "sexo", {"h":"H", "v":"V"})
        custom_one_hot(train_df, features, "ind_empleado", {"a":"A", "b":"B", "f":"F", "n":"N"})
        custom_one_hot(train_df, features, "ind_nuevo", {"new":1})
        custom_one_hot(train_df, features, "segmento", {"top":"01 - TOP", "particulares":"02 - PARTICULARES", "universitario":"03 - UNIVERSITARIO"})
        custom_one_hot(train_df, features, "indfall", {"s":"S"})

        train_df["ind_actividad_cliente"] = train_df["ind_actividad_cliente"].map(lambda x: 0.0 if math.isnan(x) else x+1.0).astype(np.int8)
        features.append("ind_actividad_cliente")
        custom_one_hot(train_df, features, "indrel", {"1":1, "99":99})
        train_df["indrel_1mes"] = train_df["indrel_1mes"].map(lambda x: 5.0 if x == "P" else x).astype(float).fillna(0.0).astype(np.int8)
        assert_uniq(train_df["indrel_1mes"], "indrel_1mes")
        features.append("indrel_1mes")
        custom_one_hot(train_df, features, "tiprel_1mes", {"a":"A", "i":"I", "p":"P", "r":"R"}, check=True)

    return train_df, tuple(features)


def make_prev_df(train_df, step):
    with Timer("make prev%s DF" % step):
        prev_df = pd.DataFrame()
        prev_df["ncodpers"] = train_df["ncodpers"]
        prev_df["int_date"] = train_df["int_date"].map(lambda x: x+step).astype(np.int8)
        prod_features = ["%s_prev%s" % (prod, step) for prod in products]
        for prod, prev in zip(products, prod_features):
            prev_df[prev] = train_df[prod]
    return prev_df, tuple(prod_features)


def load_data(fname="../input/all_clean.csv"):
    with Timer("load train csv"):
        train_df = pd.read_csv(fname, dtype=dtypes)

    with Timer("fill products NA"):
        for prod in products:
            train_df[prod] = train_df[prod].fillna(0.0).astype(np.int8)

    train_df, features = apply_transforms(train_df)

    prev_dfs = []

    prod_features = None

    use_features = frozenset([1,2])
    for step in range(1,6):
        prev1_train_df, prod1_features = make_prev_df(train_df, step)
        prev_dfs.append(prev1_train_df)
        if step in use_features:
            features += prod1_features
        if step == 1:
            prod_features = prod1_features

    return train_df, prev_dfs, features, prod_features


def join_with_prev(df, prev_df, how):
    with Timer("join %s" % how):
        assert set(df.columns.values.tolist()) & set(prev_df.columns.values.tolist()) == set(["ncodpers", "int_date"])
        print("before join", len(df))
        df = df.merge(prev_df, on=["ncodpers", "int_date"], how=how)
        for f in set(prev_df.columns.values.tolist()) - set(["ncodpers", "int_date"]):
            df[f] = df[f].astype(np.float16)
        print("after join", len(df))
        return df

def make_data():
    train_df, prev_dfs, features, prod_features = load_data()

    for i, prev_df in enumerate(prev_dfs):
        with Timer("join train with prev%s" % (i+1)):
            how = "inner" if i == 0 else "left"
            train_df = join_with_prev(train_df, prev_df, how=how)

    # Various aggregates to try
    # for prod in products:
    #     print()
    #     print(prod)
    #     #prev1_bin = (train_df[prod + "_prev1"] != 1).astype(np.int8)
    #     for begin, end in [(2,5),(1,4)]:
    #         prods = ["%s_prev%s" % (prod, i) for i in range(begin,end+1)]
    #         mp_df = train_df.as_matrix(columns=prods)
    #         print(prods)
    #
    #         stdf = "%s_std_%s_%s" % (prod,begin,end)
    #         train_df[stdf] = np.nanstd(mp_df, axis=1) #  * prev1_bin
    #
    #         maxf = "%s_max_%s_%s"%(prod,begin,end)
    #         train_df[maxf] = np.nanmax(mp_df, axis=1).astype(np.int8)
    #
    #         # minf = "%s_min_%s_%s"%(prod,begin,end)
    #         # train_df[minf] = np.nanmin(mp_df, axis=1).astype(np.int8)
    #
    #         chf = "%s_ch_%s_%s"%(prod,begin,end)
    #         train_df[chf] = np.sum(np.invert(np.isclose(mp_df[:,1:], mp_df[:,:-1], equal_nan=True)), axis=1, dtype=np.int8)
    #
    #         sumf = "%s_sum_%s_%s"%(prod,begin,end)
    #         train_df[sumf] = np.nansum(mp_df, axis=1, dtype=np.int8)
    #
    #         skewf = "%s_skew_%s_%s"%(prod,begin,end)
    #         train_df[skewf] = scipy.stats.skew(mp_df, axis=1)
    #
    #         features += (stdf,maxf,chf,sumf,skewf)


    for prod in products:
        print()
        print(prod)
        for begin, end in [(1,3),(1,5),(2,5)]:
            prods = ["%s_prev%s" % (prod, i) for i in range(begin,end+1)]
            mp_df = train_df.as_matrix(columns=prods)
            print(prods)

            stdf = "%s_std_%s_%s" % (prod,begin,end)
            train_df[stdf] = np.nanstd(mp_df, axis=1) #  * prev1_bin

            features += (stdf,)

    for prod in products:
        print()
        print(prod)
        for begin, end in [(2,3),(2,5)]:
            prods = ["%s_prev%s" % (prod, i) for i in range(begin,end+1)]
            mp_df = train_df.as_matrix(columns=prods)
            print(prods)

            minf = "%s_min_%s_%s"%(prod,begin,end)
            train_df[minf] = np.nanmin(mp_df, axis=1).astype(np.int8)

            maxf = "%s_max_%s_%s"%(prod,begin,end)
            train_df[maxf] = np.nanmax(mp_df, axis=1).astype(np.int8)

            features += (minf,maxf,)

    with Timer("Remove unused columns"):
        leave_columns = ["ncodpers", "int_date", "fecha_dato"] + list(products) + list(features)
        assert len(leave_columns) == len(set(leave_columns))
        train_df = train_df[leave_columns]

    return train_df, features, prod_features


def make_submission(f, Y_test, C):
    Y_ret = []
    with Timer("make submission"):
        f.write("ncodpers,added_products\n".encode('utf-8'))
        for c, y_test in zip(C, Y_test):
            y_prods = [(y,p,ip) for y,p,ip in zip(y_test, products, range(len(products)))]
            y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
            Y_ret.append([ip for y,p,ip in y_prods])
            y_prods = [p for y,p,ip in y_prods]
            f.write(("%s,%s\n" % (int(c), " ".join(y_prods))).encode('utf-8'))
    return Y_ret


def train_predict(all_df, features, prod_features, str_date, cv):
    test_date = date_to_int(str_date)
    train_df = all_df[all_df.int_date < test_date]
    test_df = pd.DataFrame(all_df[all_df.int_date == test_date])
    print(sorted(set(train_df.columns.values.tolist())))
    print(len(train_df.columns.values.tolist()), len(set(train_df.columns.values.tolist())))
    print(len(features),len(set(features)))

    X = []
    Y = []
    for i, prod in enumerate(products):
        prev = prod + "_prev1"
        prX = train_df[(train_df[prod] == 1) & (train_df[prev] == 0)]
        prY = np.zeros(prX.shape[0], dtype=np.int8) + i
        X.append(prX)
        Y.append(prY)
        print(prod, prX.shape)


    XY = pd.concat(X)
    Y = np.hstack(Y)
    XY["y"] = Y
    XY["url"] = np.zeros(len(XY), dtype=np.int8)

    del train_df
    del all_df


    XY["ncodepers_fecha_dato"] = XY["ncodpers"].astype(str) + XY["fecha_dato"]
    uniqs, counts = np.unique(XY["ncodepers_fecha_dato"], return_counts=True)
    weights = np.exp(1/counts - 1)
    print(np.unique(counts, return_counts=True))
    print(np.unique(weights, return_counts=True))
    wdf = pd.DataFrame()
    wdf["ncodepers_fecha_dato"] = uniqs
    wdf["counts"] = counts
    wdf["weight"] = weights
    print("before merge", len(XY))
    XY = XY.merge(wdf, on="ncodepers_fecha_dato")
    print("after merge", len(XY))

    print(XY.shape)

    mask = np.random.rand(len(XY)) < 0.8
    XY_train = XY[mask]
    XY_validate = XY[~mask]

    with Timer("prepare test data"):
        test_df["y"] = test_df["ncodpers"]
        test_df["url"] = np.zeros(len(test_df), dtype=np.int8)
        test_df["weight"] = np.ones(len(test_df), dtype=np.int8)
        Y_prev = test_df.as_matrix(columns=prod_features)
        C = test_df.as_matrix(columns=["ncodpers"])
        for prod in products:
            prev = prod + "_prev1"
            padd = prod + "_add"
            test_df[padd] = test_df[prod] - test_df[prev]
        test_add_mat = test_df.as_matrix(columns=[prod + "_add" for prod in products])
        test_add_list = [list() for i in range(len(C))]
        assert test_add_mat.shape == (len(C), len(products))
        count = 0
        for c in range(len(C)):
            for p in range(len(products)):
                if test_add_mat[c,p] > 0:
                    test_add_list[c].append(p)
                    count += 1


    if cv:
        max_map7 = mapk(test_add_list, test_add_list, 7, 0.0)
        map7coef = float(len(test_add_list)) / float(sum([int(bool(a)) for a in test_add_list]))
        print("Max MAP@7", str_date, max_map7, max_map7*map7coef)

    with Timer("LightGBM"):
        Y_test_lgbm = engines.lightgbm(XY_train, XY_validate, test_df, features, XY_all = XY,
            restore = (str_date == "2016-06-28")
        )
        test_add_list_lightgbm = make_submission(io.BytesIO() if cv else gzip.open("tmp/%s.lightgbm.csv.gz" % str_date, "wb"),
                                                  Y_test_lgbm - Y_prev, C)
        if cv:
            map7lightgbm = mapk(test_add_list, test_add_list_lightgbm, 7, 0.0)
            print("LightGBMlib MAP@7", str_date, map7lightgbm, map7lightgbm*map7coef)

    with Timer("XGBoost"):
        Y_test_xgb = engines.xgboost(XY_train, XY_validate, test_df, features, XY_all = XY,
            restore = (str_date == "2016-06-28")
        )
        test_add_list_xgboost = make_submission(io.BytesIO() if cv else gzip.open("tmp/%s.xgboost.csv.gz" % str_date, "wb"),
                                                Y_test_xgb - Y_prev, C)
        if cv:
            map7xgboost = mapk(test_add_list, test_add_list_xgboost, 7, 0.0)
            print("XGBoost MAP@7", str_date, map7xgboost, map7xgboost*map7coef)

    Y_test = np.sqrt(np.multiply(Y_test_xgb, Y_test_lgbm))

    test_add_list_xl = make_submission(io.BytesIO() if cv else gzip.open("tmp/%s.xgboost-lightgbm.csv.gz" % str_date, "wb"),
                                       Y_test - Y_prev, C)
    if cv:
        map7xl = mapk(test_add_list, test_add_list_xl, 7, 0.0)
        print("XGBoost+LightGBM MAP@7", str_date, map7xl, map7xl*map7coef)




if __name__ == "__main__":
    if True:
        all_df, features, prod_features = make_data()
        with Timer("save data"):
            all_df.to_pickle("tmp/cv_data.pickle")
            pickle.dump((features, prod_features), open("tmp/cv_meta.pickle", "wb"))
    else:
        with Timer("restore data"):
            all_df = pd.read_pickle("tmp/cv_data.pickle")
            (features, prod_features) = pickle.load(open("tmp/cv_meta.pickle", "rb"))



    train_predict(all_df, features, prod_features, "2016-05-28", cv=True)
    train_predict(all_df, features, prod_features, "2016-06-28", cv=False)
