from fault_diagnosis_baseline import fdob
from fault_diagnosis_baseline.fdob import processing
from torchvision import transforms
# from cProfile import label
import wget
import zipfile
from scipy import io
import numpy as np
import pandas as pd
import os
import patoolib

def get_datamodule(data_path: str, sample_length: int, batch_size: int, num_workers: int, tf_data, tf_label, snr_values, sample_shift: int=None):

    sample_shift = sample_shift if sample_shift else sample_length // 2

    if data_path.endswith('/'):
        data_path = data_path[:-1]

    dataset_name = data_path.split('/')[-1]
    if dataset_name == 'cwru48k':
        dataset_name = 'cwru'
        download_dataset = getattr(fdob, f'download_{dataset_name}')
        df = download_dataset(data_path, sample_rate='48k')
    else:
        download_dataset = getattr(fdob, f'download_{dataset_name}')

        df = download_dataset(data_path)
    # We exclude label named 999 and 0 HP motor load condition.
    if dataset_name in ['cwru', 'mfpt']:
        df = df[(df["label"] != 999) & (df["load"] != 0)]

    train_df, val_df, test_df = fdob.split_dataframe(df, 0.6, 0.2)

    X_train, y_train = fdob.build_from_dataframe(train_df, sample_length, sample_shift, False)
    X_val, y_val = fdob.build_from_dataframe(val_df, sample_length, sample_shift, False)
    X_test, y_test = fdob.build_from_dataframe(test_df, sample_length, sample_shift, False)

    dmodule = fdob.DatasetHandler()

    dmodule.assign(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        sample_length,
        dataset_name,
        transforms.Compose(tf_data),
        transforms.Compose(tf_label),
        batch_size,
        num_workers
    )

    for snr in snr_values:
        dmodule.assign(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            sample_length,
            f"{dataset_name}{snr}",
            transforms.Compose([processing.AWGN(snr)] + tf_data),
            transforms.Compose(tf_label),
            batch_size,
            num_workers
        )

    return dmodule


def download_mfpt(root: str, downsample='none') -> pd.DataFrame:
    """
    Download the MFPT dataset.
    Author: Seongjae Lee

    Parameters
    ----------
    root: str
        Root directory where the data files are saved.

    Returns
    ----------
    pd.DataFrame
        Return dataframe containing data segments of the MFPT dataset.
    """
    url = "https://www.mfpt.org/wp-content/uploads/2020/02/MFPT-Fault-Data-Sets-20200227T131140Z-001.zip"
    zipname = "data.zip"
    datafolder = "MFPT Fault Data Sets"

    assert downsample in ['none', '12k', '48k'], "Argument 'downsample' should be one of 'none', '12k' or '48k'."

    filenames = [
        (f"{root}/{datafolder}/1 - Three Baseline Conditions/baseline_1.mat", "N"),
        (f"{root}/{datafolder}/1 - Three Baseline Conditions/baseline_2.mat", "N"),
        (f"{root}/{datafolder}/1 - Three Baseline Conditions/baseline_3.mat", "N"),
        (
            f"{root}/{datafolder}/2 - Three Outer Race Fault Conditions/OuterRaceFault_1.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/2 - Three Outer Race Fault Conditions/OuterRaceFault_2.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/2 - Three Outer Race Fault Conditions/OuterRaceFault_3.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_1.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_2.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_3.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_4.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_5.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_6.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/3 - Seven More Outer Race Fault Conditions/OuterRaceFault_vload_7.mat",
            "OR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_1.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_2.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_3.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_4.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_5.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_6.mat",
            "IR",
        ),
        (
            f"{root}/{datafolder}/4 - Seven Inner Race Fault Conditions/InnerRaceFault_vload_7.mat",
            "IR",
        ),
    ]

    label_map = {"N": 0, "IR": 1, "OR": 2}

    if not os.path.isdir(root):
        os.makedirs(root)

    if not os.path.isdir(f"{root}/{datafolder}"):
        os.system(f"wget -O {root}/{zipname} {url}")
        with zipfile.ZipFile(f"{root}/{zipname}", "r") as f:
            f.extractall(f"{root}")
        os.remove(f"{root}/{zipname}")
    else:
        print("File is already existed, use existed file.")

    df = {}
    df["data"] = []
    df["fault_type"] = []
    df["sampling_rate"] = []
    df["load"] = []
    df["shaft_rate"] = []
    df["label"] = []

    for file in filenames:
        filename = file[0]
        fault_type = file[1]
        data = io.loadmat(filename)
        sr = data["bearing"]["sr"][0][0].ravel()[0]
        body = data["bearing"]["gs"][0][0].ravel()
        load = data["bearing"]["load"][0][0].ravel()[0]
        shaft_rate = data["bearing"]["rate"][0][0].ravel()[0]

        label = label_map[fault_type]


        if not downsample == 'none':
            target_sr = 12207 if downsample == '12k' else 48828
            stride = sr // target_sr

            if stride > 1:
                sr = target_sr
                body = body[0::stride]


        df["fault_type"].append(fault_type)
        df["data"].append(body)
        df["sampling_rate"].append(sr)
        df["load"].append(load)
        df["shaft_rate"].append(shaft_rate)
        df["label"].append(label)

    data_frame = pd.DataFrame(df)

    return data_frame


def get_mfpt_dm(data_path: str, sample_length: int, batch_size: int, num_workers: int, tf_data, tf_label, snr_values, sample_shift: int=None, downsample='12k'):

    sample_shift = sample_shift if sample_shift else sample_length // 2

    df = download_mfpt(data_path, downsample=downsample)
    
    df = df[(df["label"] != 999) & (df["load"] != 0)]

    train_df, val_df, test_df = fdob.split_dataframe(df, 0.6, 0.2)

    X_train, y_train = fdob.build_from_dataframe(train_df, sample_length, sample_shift, False)
    X_val, y_val = fdob.build_from_dataframe(val_df, sample_length, sample_shift, False)
    X_test, y_test = fdob.build_from_dataframe(test_df, sample_length, sample_shift, False)

    tag = f"mfpt_{downsample}hz"

    dmodule = fdob.DatasetHandler()

    dmodule.assign(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        sample_length,
        tag,
        transforms.Compose(tf_data),
        transforms.Compose(tf_label),
        batch_size,
        num_workers
    )

    for snr in snr_values:
        dmodule.assign(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            sample_length,
            f"{tag}{snr}",
            transforms.Compose([processing.AWGN(snr)] + tf_data),
            transforms.Compose(tf_label),
            batch_size,
            num_workers
        )

    return dmodule