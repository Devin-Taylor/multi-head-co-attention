import json
import os

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = "./"
DATA = os.path.join(ROOT, "data/")

def load_ppmi(dataset, filter_feats=True, normalise=True):
    patient_summary = pd.read_csv(os.path.join(DATA, "patient_summary.csv"))
    metadata = patient_summary[patient_summary.DIAGNOSIS == dataset]
    methylation_data_path = os.path.join(DATA, "ppmi_only_caucasian_Zhou_filtered.pkl")
    meth_data = pd.read_pickle(methylation_data_path)

    metadata = pd.merge(metadata, meth_data, how="left", left_on="meth_file", right_index=True).dropna().reset_index(drop=True)

    spect_root = "CONTROLS" if dataset == "CONTROL" else "PD"

    spect_data = np.array([np.asarray(nib.load(os.path.join(DATA, spect_root, x)).dataobj) for x in metadata.spect_file])
    spect_data = np.expand_dims(spect_data, axis=1) # insert channel
    meth_data = metadata.iloc[:, 5:] # NOTE hard indexing so keep this in mind
    metadata = metadata.iloc[:, :5]

    with open(os.path.join(DATA, "meth_classification_filtered_xgb_tuned_441.json")) as fd:
        meth_features = json.load(fd)

    if filter_feats:
        meth_data = meth_data.loc[:, meth_data.columns.isin(meth_features)]

    if normalise:
        # NOTE these values are the max and min across all control and PD images in the database
        global_min = -1.0
        global_max = 5.6598325
        spect_data = (spect_data - global_min)/(global_max - global_min)

    age = metadata.Age.values
    return meth_data, spect_data, age, metadata
