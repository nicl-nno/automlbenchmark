import logging
import sys

import os
import tempfile as tmp

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from fedot.api.main import Fedot

from frameworks.shared.callee import call_run, result, output_subdir, utils

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** FEDOT ****\n")

    is_classification = config.type == 'classification'
    # Mapping of benchmark metrics to FEDOT metrics
    metrics_mapping = dict(
        acc='acc',
        auc='roc_auc',
        f1='f1',
        logloss='logloss',
        mae='mae',
        mse='mse',
        msle='msle',
        r2='r2',
        rmse='rmse'
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None

    if scoring_metric is None:
        raise ValueError(f'Performance metric {config.metric} not supported.')

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    n_jobs = config.framework_params.get('_n_jobs',
                                         config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info('Running FEDOT with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)
    runtime_min = (config.max_runtime_seconds / 60)

    fedot = Fedot(problem=config.type, learning_time=runtime_min,
                  composer_params={'metric': scoring_metric}, **training_params)

    with utils.Timer() as training:
        model = fedot.fit(features=dataset.train.X_enc, target=dataset.train.y_enc)
        #if config.type == 'classification':
        #    model = fedot.fit(features=dataset.train.X_enc, target=dataset.train.y_enc, predefined_model='logit')
        #else:
        #    model = fedot.fit(features=dataset.train.X_enc, target=dataset.train.y_enc, predefined_model='linear')

    log.info('Predicting on the test set.')
    predictions = fedot.predict(features=dataset.test.X_enc)
    if config.type == 'classification':
        predictions = fedot.prediction_labels.predict

    if not is_classification:
        probabilities = None
    else:
        probabilities = fedot.predict_proba(features=dataset.test.X_enc, probs_for_all_classes=True)

    save_artifacts({'model': model}, config)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=dataset.test.y_enc,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=1,
                  training_duration=training.duration)


def save_artifacts(chain, config):
    try:
        artifacts = config.framework_params.get('_save_artifacts', False)
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'model.json')
            chain.save(models_file)
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
