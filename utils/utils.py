import argparse
import os
import time

import pandas as pd
import torch
from pytorch_lightning import seed_everything

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping

from log import logger
from model.ner_model import NERBaseAnnotator
from model.ner_model_sum_align_tok import NERAlignv1Annotator
from model.ner_model_cls_align import NERAlignv2Annotator
from model.ner_align_tok_trans import NERAlignv3Annotator
from model.ner_model_concat import NERBaseAnnotatorv4
from model.ner_model_v5 import NERAlignv5Annotator
from utils.reader import CoNLLReader

conll_iob = {'B-ORG': 0, 'I-ORG': 1, 'B-MISC': 2, 'I-MISC': 3, 'B-LOC': 4, 'I-LOC': 5, 'B-PER': 6, 'I-PER': 7, 'O': 8}
wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7, 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
resume_iob = {'M-RACE': 0, 'B-PRO': 1, 'S-ORG': 2, 'B-LOC': 3, 'B-CONT': 4, 'M-CONT': 5, 'E-LOC': 6, 'M-PRO': 7, 'M-LOC': 8, 'M-TITLE': 9, 'B-ORG': 10, 'M-ORG': 11, 'E-ORG': 12,
              'E-RACE': 13, 'B-EDU': 14, 'S-NAME': 15, 'B-TITLE': 16, 'S-RACE': 17, 'B-NAME': 18, 'B-RACE': 19, 'E-NAME': 20, 'O': 21, 'E-CONT': 22, 'M-EDU': 23, 'E-TITLE': 24, 'E-EDU': 25,
              'M-NAME': 26, 'E-PRO': 27}
weibo_iob = {'O': 0, 'B-PER.NOM': 1, 'E-PER.NOM': 2, 'B-LOC.NAM': 3, 'E-LOC.NAM': 4, 'B-PER.NAM': 5, 'M-PER.NAM': 6, 'E-PER.NAM': 7, 'S-PER.NOM': 8, 'B-GPE.NAM': 9, 'E-GPE.NAM': 10,
             'B-ORG.NAM': 11, 'M-ORG.NAM': 12, 'E-ORG.NAM': 13, 'M-PER.NOM': 14, 'S-GPE.NAM': 15, 'B-ORG.NOM': 16, 'E-ORG.NOM': 17, 'M-LOC.NAM': 18, 'M-ORG.NOM': 19, 'B-LOC.NOM': 20,
             'M-LOC.NOM': 21, 'E-LOC.NOM': 22, 'B-GPE.NOM': 23, 'E-GPE.NOM': 24, 'M-GPE.NAM': 25, 'S-PER.NAM': 26, 'S-LOC.NOM': 27}
msra_iob = {'O': 0, 'S-NS': 1, 'B-NS': 2, 'E-NS': 3, 'B-NT': 4, 'M-NT': 5, 'E-NT': 6, 'M-NS': 7, 'B-NR': 8, 'M-NR': 9, 'E-NR': 10, 'S-NR': 11, 'S-NT': 12}
ontonotes_iob = {'E-PER': 0, 'E-GPE': 1, 'E-LOC': 2, 'M-ORG': 3, 'E-ORG': 4, 'S-ORG': 5, 'B-GPE': 6, 'O': 7, 'M-PER': 8, 'M-LOC': 9, 'B-PER': 10, 'M-GPE': 11, 'S-LOC': 12, 'B-ORG': 13,
                 'S-PER': 14, 'B-LOC': 15, 'S-GPE': 16}
multiconerii_iob = {'I-Food': 40, 'I-Athlete': 16, 'I-MusicalGRP': 29, 'B-Facility': 20, 'B-Politician': 13, 'I-AerospaceManufacturer': 65, 'B-Vehicle': 50, 'I-Vehicle': 51,
                    'I-HumanSettlement': 19, 'B-Food': 39, 'I-Software': 11, 'B-MusicalWork': 17, 'I-Station': 42, 'I-Cleric': 25, 'I-Medication/Vaccine': 55, 'I-SportsGRP': 48,
                     'I-Drink': 63, 'B-ArtWork': 37, 'B-PublicCorp': 32, 'I-Artist': 7, 'I-OtherPROD': 43, 'B-ORG': 5, 'B-SportsManager': 30, 'I-ORG': 26, 'I-Politician': 14,
                     'B-Cleric': 24, 'I-CarManufacturer': 61, 'B-Artist': 6, 'B-WrittenWork': 9, 'I-Disease': 56, 'B-Disease': 49, 'B-Athlete': 15, 'I-PrivateCorp': 52, 'I-OtherLOC': 46,
                     'B-OtherPER': 0, 'I-ArtWork': 38, 'B-Scientist': 22, 'B-MedicalProcedure': 35, 'B-Drink': 62, 'I-Facility': 21, 'B-AnatomicalStructure': 57, 'O': 2, 'I-MedicalProcedure': 36,
                     'B-Medication/Vaccine': 53, 'I-SportsManager': 31, 'I-AnatomicalStructure': 58, 'I-Clothing': 66, 'I-Symptom': 59, 'B-HumanSettlement': 8, 'I-Scientist': 23, 'B-Software': 10,
                     'B-SportsGRP': 27, 'I-PublicCorp': 33, 'B-CarManufacturer': 44, 'I-WrittenWork': 12, 'B-Symptom': 54, 'B-AerospaceManufacturer': 60, 'B-OtherPROD': 34, 'I-OtherPER': 1,
                     'B-VisualWork': 3, 'B-PrivateCorp': 47, 'B-MusicalGRP': 28, 'I-MusicalWork': 18, 'B-Station': 41, 'B-Clothing': 64, 'B-OtherLOC': 45, 'I-VisualWork': 4, 'B-OtherCW': 67,
                    'I-OtherCW': 68, 'B-OtherCorp': 69, 'I-OtherCorp': 70, 'B-TechCORP': 71, 'I-TechCORP':72 }

def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=False)
    p.add_argument('--train', type=str, help='Path to the train data.', default=None)
    p.add_argument('--test', type=str, help='Path to the test data.', default=None)
    p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)

    p.add_argument('--out_dir', type=str, help='Output directory.', default='.')
    p.add_argument('--iob_tagging', type=str, help='IOB tagging scheme', default='multiconer')

    p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=-1)
    p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=50)

    p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
    p.add_argument('--model', type=str, help='Model path.', default=None)
    p.add_argument('--model_name', type=str, help='Model name.', default=None)
    p.add_argument('--stage', type=str, help='Training stage', default='fit')
    p.add_argument('--prefix', type=str, help='Prefix for storing evaluation files.', default='test')

    p.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
    p.add_argument('--cuda', type=str, help='Cuda Device', default='cuda:0')
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
    p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)

    p.add_argument('--b', type=float, help='Value of b hyperparam', default=0.15)
    p.add_argument('--lang', type=str, help='Fix the definitions language', default='EN_1')
    p.add_argument('--seed', type=int, help='Random seed', default=42)

    p.add_argument('--optuna_db', type=str, help='Optuna db name', default='example.db')
    p.add_argument('--optuna_name', type=str, help='Study name', default='example-study')
    p.add_argument('--version_model', type=str, help='Version model tag', default='v0')
    
    return p.parse_args()


def get_tagset(tagging_scheme):
    if os.path.isfile(tagging_scheme):
        # read the tagging scheme from a file
        sep = '\t' if tagging_scheme.endswith('.tsv') else ','
        df = pd.read_csv(tagging_scheme, sep=sep)
        tags = {row['tag']: row['idx'] for idx, row in df.iterrows()}
        return tags

    if 'multiconer' in tagging_scheme:
        return multiconerii_iob
    if 'conll' in tagging_scheme:
        return conll_iob
    elif 'wnut' in tagging_scheme:
        return wnut_iob
    elif 'resume' in tagging_scheme:
        return resume_iob
    elif 'ontonotes' in tagging_scheme:
        return ontonotes_iob
    elif 'msra' in tagging_scheme:
        return msra_iob
    elif 'weibo' in tagging_scheme:
        return weibo_iob


def get_out_filename(out_dir, model, prefix):
    model_name = os.path.basename(model)
    model_name = model_name[:model_name.rfind('.')]
    return '{}/{}_base_{}.tsv'.format(out_dir, prefix, model_name)


def write_eval_performance(eval_performance, out_file):
    outstr = ''
    added_keys = set()
    for out_ in eval_performance:
        for k in out_:
            if k in added_keys or k in ['results', 'predictions']:
                continue
            outstr = outstr + '{}\t{}\n'.format(k, out_[k])
            added_keys.add(k)

    open(out_file, 'wt').write(outstr)
    logger.info('Finished writing evaluation performance for {}'.format(out_file))


def get_reader(file_path, max_instances=-1, max_length=50, target_vocab=None, encoder_model='xlm-roberta-large',
               train=False, lang='EN_1'):
    if file_path is None:
        return None
    reader = CoNLLReader(max_instances=max_instances, max_length=max_length, target_vocab=target_vocab, encoder_model=encoder_model)
    reader.read_data(file_path, train=train, lang=lang)

    return reader


def create_model(train_data, dev_data, tag_to_id, batch_size=64, dropout_rate=0.1, stage='fit', lr=1e-5,
                 encoder_model='xlm-roberta-large', num_gpus=1, model_type='v0', b=0.0):
    if model_type=='v0':
        print("Model", model_type)
        return NERBaseAnnotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model,
                            dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus)
    elif model_type=='v1':
        return NERAlignv1Annotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model,
                                   dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id,
                                   num_gpus=num_gpus, b=b)
    # elif model_type=='v2':
    #     return NERAlignv2Annotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model, dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus)
    # elif model_type=='v3':
    #     return NERAlignv3Annotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model, dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus)
    # elif model_type=='v4':
    #     return NERBaseAnnotatorv4(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model, dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus)
    # elif model_type=='v5':
    #     return NERAlignv5Annotator(train_data=train_data, dev_data=dev_data, tag_to_id=tag_to_id, batch_size=batch_size, stage=stage, encoder_model=encoder_model, dropout_rate=dropout_rate, lr=lr, pad_token_id=train_data.pad_token_id, num_gpus=num_gpus)
    #
        

def load_model(model_file, tag_to_id=None, stage='test', model_type='v0'):
    if ~os.path.isfile(model_file):
        model_file = get_models_for_evaluation(model_file)

    hparams_file = model_file[:model_file.rindex('checkpoints/')] + '/hparams.yaml'
    if model_type=='v0':
        model = NERBaseAnnotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    elif model_type=='v1':
        model = NERAlignv1Annotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    # elif model_type=='v2':
    #     model = NERAlignv2Annotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    # elif model_type=='v3':
    #     model = NERAlignv3Annotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    # elif model_type=='v4':
    #     model = NERBaseAnnotatorv4.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    # elif model_type=='v5':
    #     model = NERAlignv5Annotator.load_from_checkpoint(model_file, hparams_file=hparams_file, stage=stage, tag_to_id=tag_to_id)
    model.stage = stage
    return model, model_file


def save_model(trainer, out_dir, model_name='', timestamp=None):
    out_dir = out_dir + '/lightning_logs/version_' + str(trainer.logger.version) + '/checkpoints/'
    if timestamp is None:
        timestamp = time.time()
    os.makedirs(out_dir, exist_ok=True)

    outfile = out_dir + '/' + model_name + '_timestamp_' + str(timestamp) + '_final.ckpt'
    trainer.save_checkpoint(outfile, weights_only=True)

    logger.info('Stored model {}.'.format(outfile))
    return outfile


# def get_trainer(gpus=4, is_test=False, out_dir=None, epochs=10):
#     seed_everything(42)
#     if is_test:
#         return pl.Trainer(gpus=1) if torch.cuda.is_available() else pl.Trainer(val_check_interval=100)

#     if torch.cuda.is_available():
#         trainer = pl.Trainer(gpus=gpus, deterministic=False, max_epochs=epochs, callbacks=[get_model_earlystopping_callback()],
#                              default_root_dir=out_dir, strategy='ddp', checkpoint_callback=False)
#         trainer.callbacks.append(get_lr_logger())
#     else:
#         trainer = pl.Trainer(max_epochs=epochs, default_root_dir=out_dir)

#     return trainer



def train_model(model, out_dir='', epochs=10, gpus=1, trial=None, seed=42):
    trainer = get_trainer(gpus=gpus, out_dir=out_dir, epochs=epochs, trial=trial, seed=seed)
    trainer.fit(model)
    return trainer


def get_trainer(gpus=4, is_test=False, out_dir=None, epochs=10, trial=None, seed=42):
    seed_everything(seed)
    if is_test:
        return pl.Trainer(gpus=1) if torch.cuda.is_available() else pl.Trainer(val_check_interval=100)

    if torch.cuda.is_available():
        if trial is None:
            print("Here is correct")
            trainer = pl.Trainer(gpus=gpus, deterministic=False, max_epochs=epochs, 
                                 #callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_micro@F1")],
                                 callbacks=[get_model_earlystopping_callback()],
                                 default_root_dir=out_dir, strategy='ddp', 
                                 checkpoint_callback=False, enable_checkpointing=False)
        else:
            trainer = pl.Trainer(gpus=gpus, deterministic=False, max_epochs=epochs, 
                                 callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
                                 #callbacks=[get_model_earlystopping_callback()],
                                 default_root_dir=out_dir, strategy='ddp', 
                                 checkpoint_callback=False, enable_checkpointing=False)
        trainer.callbacks.append(get_lr_logger())
    else:
        trainer = pl.Trainer(max_epochs=epochs, default_root_dir=out_dir)

    return trainer


def get_lr_logger():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    return lr_monitor


def get_model_earlystopping_callback():
    es_clb = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=True,
        mode='min'
    )
    #es_clb = EarlyStopping(
    #    monitor='val_micro@F1',
    #    min_delta=0.1,
    #    patience=2,
    #    verbose=True,
    #    mode='max'
    #)
    return es_clb


def get_models_for_evaluation(path):
    if 'checkpoints' not in path:
        path = path + '/checkpoints/'
    model_files = list_files(path)
    models = [f for f in model_files if f.endswith('final.ckpt')]

    return models[0] if len(models) != 0 else None


def list_files(in_dir):
    files = []
    for r, d, f in os.walk(in_dir):
        for file in f:
            files.append(os.path.join(r, file))
    return files


from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
import optuna

class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
