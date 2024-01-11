from fault_diagnosis_baseline import fdob, info
from fault_diagnosis_baseline.fdob import processing
from fault_diagnosis_baseline.fdob.download import download_ottawa
from fault_diagnosis_baseline.fdob.model.module import Conv1d, Conv2d
from pruning.utils import LightningModuleWrapper
from pruning.activation.structured import OutputFeaturemapPrune
from pruning.activation.unstructured import OutputActivationPrune

from argparse import ArgumentParser, Namespace
from torch.optim import Adam
from pytorch_lightning.callbacks import ModelCheckpoint

import random
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import pytorch_lightning as pl

import mlflow

import logging

n_classes_map = {
    'cwru': 10,
    'mfpt': 3,
    'ottawa': 5,
}

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--datapath',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model-name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--seed',
        nargs='+',
        type=int,
        required=True
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100
    )
    parser.add_argument(
        '--activation-drop',
        choices=['activation', 'featuremap', 'none']
    )
    parser.add_argument(
        '--score-type',
        choices=['max', 'l1', 'l2']
    )
    parser.add_argument(
        '--activation-sparsity',
        type=float,
        default=None
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128
    )
    parser.add_argument(
        '--snr-values',
        nargs='+',
        type=int,
        default=[0]
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0001
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0
    )
    return parser.parse_args()


def main(seed, args: Namespace):

    logging.getLogger().setLevel(logging.INFO)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    model_name = args.model_name
    data_path = args.datapath
    batch_size = args.batch_size
    num_workers = args.num_workers
    snr_values = args.snr_values
    drop_method = args.activation_drop
    score_type = args.score_type
    sparsity = args.activation_sparsity

    sample_shift = 1024

    cuda_deterministic = True

    # For reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = cuda_deterministic
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    accelerator = 'gpu'

    model_cls = info.model[model_name]["model"]
    sample_length = info.model[model_name]["sample_length"]
    tf_data = info.model[model_name]["tf"]
    if not model_name == 'stftcnn':
        tf_data += [processing.InfinityNorm()]
    tf_label = [processing.NpToTensor()]

    df = download_ottawa(data_path)

    def downsample(data: np.ndarray):
        return data[::4]
    
    df['data'] = df['data'].apply(downsample)

    train_df, val_df, test_df = fdob.split_dataframe(df, 0.6, 0.2)

    X_train, y_train = fdob.build_from_dataframe(train_df, sample_length, sample_shift, False)
    X_val, y_val = fdob.build_from_dataframe(val_df, sample_length, sample_shift, False)
    X_test, y_test = fdob.build_from_dataframe(test_df, sample_length, sample_shift, False)

    tag = "ottawa"

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

    train_loader = dmodule.dataloaders[tag]['train']
    val_loader = dmodule.dataloaders[tag]['val']
    test_loader = dmodule.dataloaders[tag]['test']

    n_classes = n_classes_map['ottawa']

    if drop_method == 'featuremap':
        fm_prune = OutputFeaturemapPrune(sparsity=sparsity, score_type=score_type)
        model = model_cls(n_classes=n_classes, act_layer=False, no_drop=True)

        for name, module in model.named_modules():
            if isinstance(module, (Conv1d, Conv2d)):
                logging.info(f"{name} will be pruned with {sparsity} sparsity.")
                fm_prune.apply(module)
    
    elif drop_method == 'activation':
        act_prune = OutputActivationPrune(sparsity=sparsity)
        model = model_cls(n_classes=n_classes, act_layer=False, no_drop=True)

        for name, module in model.named_modules():
            if isinstance(module, (Conv1d, Conv2d)):
                logging.info(f"{name} will be pruned with {sparsity} sparsity.")
                act_prune.apply(module)

    else:
        logging.info("The network will not be pruned.")
        model = model_cls(n_classes=n_classes, act_layer=True)

    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    pl_module = LightningModuleWrapper(model=model, optimizer=optimizer)

    callbacks = []
    
    experiment_name = f'STIM-CNN OTTAWA 12kHz'

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
    else:
        experiment_id = mlflow.create_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id):
        
        params = vars(args)
        params['seed'] = seed
        mlflow.log_params(params)

        artifact_uri = mlflow.get_artifact_uri()
        if artifact_uri.startswith('file:'):
            artifact_path = artifact_uri[5:]

        checkpoint_callback = ModelCheckpoint(
            dirpath=artifact_path,
            save_last=True
        )
        callbacks.append(checkpoint_callback)

        trainer = pl.Trainer(
            check_val_every_n_epoch=1,
            callbacks=callbacks,
            accelerator=accelerator,
            max_epochs=args.epochs,
            deterministic=cuda_deterministic,
            benchmark=False,
            log_every_n_steps=len(train_loader),
            enable_progress_bar=False
        )

        mlflow.pytorch.autolog(log_every_n_epoch=1, log_models=False)

        trainer.fit(
            pl_module,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )

        mlflow.pytorch.autolog(disable=True)

        ret = trainer.test(
            pl_module,
            dataloaders=test_loader
        )
        for key, value in ret[-1].items():
            mlflow.log_metric(f'{key}_ottawa', value)

        for snr in snr_values:
            tag = f'ottawa{snr}'
            noisy_loader = dmodule.dataloaders[tag]['test']
            ret = trainer.test(
                pl_module,
                dataloaders=noisy_loader
            )
            for key, value in ret[-1].items():
                mlflow.log_metric(f'{key}_{tag}', value)
        

if __name__ == '__main__':
    args = parse_args()

    for seed in args.seed:
        main(seed, args)