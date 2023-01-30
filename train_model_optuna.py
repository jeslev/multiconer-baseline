import time

from utils.utils import get_reader, train_model, create_model, save_model, parse_args, get_tagset

import os

import optuna
from optuna.trial import TrialState


sg = parse_args()

def objective(trial):
    timestamp = time.time()
    
    out_dir_path = sg.out_dir + '/' + sg.model_name

    # load the dataset first
    train_data = get_reader(file_path=sg.train, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length)
    dev_data = get_reader(file_path=sg.dev, target_vocab=get_tagset(sg.iob_tagging), encoder_model=sg.encoder_model, max_instances=sg.max_instances, max_length=sg.max_length)

    
    dropout = trial.suggest_uniform("dropout", 0.2, 0.5)
    batch_size=trial.suggest_categorical('batch',[16,32,64])
    lr = trial.suggest_float("lr", 1e-7, 0.1, log=True)
    print("Params Dropout", dropout, " Batch size", batch_size, " LR:", lr)
    model = create_model(train_data=train_data, dev_data=dev_data, tag_to_id=train_data.get_target_vocab(),
                         dropout_rate=dropout, batch_size=batch_size, stage=sg.stage, lr=lr,
                         encoder_model=sg.encoder_model, num_gpus=sg.gpus)

    
    trainer = train_model(model=model, out_dir=out_dir_path, epochs=sg.epochs, trial=trial)

    # use pytorch lightnings saver here.
    out_model_path = save_model(trainer=trainer, out_dir=out_dir_path, model_name=sg.model_name, timestamp=timestamp)
    
    return trainer.callback_metrics['val_micro@F1'].item()

if __name__ == '__main__':
    
    study = optuna.create_study(study_name=sg.optuna_name,#'multiconer-mpnet',
                                storage=f"sqlite:///{sg.optuna_db}",#'sqlite:///rob_ner.db',
                                load_if_exists=True,
                                direction="maximize")
    study.optimize(objective, n_trials=10,timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
