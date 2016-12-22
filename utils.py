import math
import time

import numpy as np

class Timer:
    def __init__(self, text=None):
        self.text = text

    def __enter__(self):
        self.cpu = time.clock()
        self.time = time.time()
        if self.text:
            print("{}...".format(self.text))
        return self

    def __exit__(self, *args):
        self.cpu = time.clock() - self.cpu
        self.time = time.time() - self.time
        if self.text:
            print("%s: cpu %0.2f, time %0.2f\n" % (self.text, self.cpu, self.time))

def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")] # "2016-05-28"
    int_date = (int(Y) - 2015) * 12 + int(M)
    assert 1 <= int_date <= 12 + 6
    return int_date

# "2016-05-28" or "" or nan
def date_to_float(str_date):
    if str_date.__class__ is float and math.isnan(str_date) or str_date == "":
        return np.nan
    Y, M, D = [int(a) for a in str_date.strip().split("-")]
    float_date = float(Y) * 12 + float(M)
    return float_date


products = (
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1" ,
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1" ,
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1" ,
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1" ,
    "ind_nomina_ult1"  ,
    "ind_nom_pens_ult1",
    "ind_recibo_ult1"  ,
)

dtypes = {
    "fecha_dato": str,
    "ncodpers": int,
    "conyuemp": str, # Spouse index. 1 if the customer is spouse of an employee
}


def apk(actual, predicted, k=10, default=1.0):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return default

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10, default=1.0):
    return np.mean([apk(a,p,k,default) for a,p in zip(actual, predicted)])
