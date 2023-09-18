from fault_diagnosis_baseline import fdob
from fault_diagnosis_baseline.fdob import processing
from torchvision import transforms

def get_datamodule(data_path: str, sample_length: int, batch_size: int, num_workers: int, tf_data, tf_label, snr_values, sample_shift: int=None):

    sample_shift = sample_shift if sample_shift else sample_length // 2

    if data_path.endswith('/'):
        data_path = data_path[:-1]

    dataset_name = data_path.split('/')[-1]

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