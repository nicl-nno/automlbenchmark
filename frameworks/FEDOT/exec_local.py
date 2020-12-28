import logging
import os
import pprint
import sys
import tempfile as tmp
from copy import copy

if sys.platform == 'darwin':
    os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
os.environ['JOBLIB_TEMP_FOLDER'] = tmp.gettempdir()
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from fedot.core.composer.gp_composer.gp_composer import GPComposer, GPComposerRequirements
from fedot.core.repository.model_types_repository import ModelTypesRepository
from fedot.core.repository.quality_metrics_repository import ClassificationMetricsEnum, RegressionMetricsEnum, \
    MetricsRepository
from fedot.core.composer.gp_composer.gp_composer import GPComposerBuilder, GPComposerRequirements
from fedot.core.composer.optimisers.gp_optimiser import GPChainOptimiserParameters, GeneticSchemeTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.data.data import InputData
import numpy as np
import datetime

import importlib.util
import json
import logging
import os
import re
import sys
from sklearn.utils import shuffle
import numpy as np
from fedot.core.chains.node import PrimaryNode
from fedot.core.chains.chain import Chain

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


amlb_path = os.environ.get("AMLB_PATH")
if amlb_path:
    utils = load_module("amlb.utils", os.path.join(amlb_path, "utils", "__init__.py"))
    NS = utils.Namespace
    touch = utils.touch
else:
    from amlb.utils import Namespace as NS, touch


def setup_logger():
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    handlers = [console]
    logging.basicConfig(handlers=handlers)
    root = logging.getLogger()
    root.setLevel(logging.INFO)


setup_logger()

log = logging.getLogger(__name__)


def result(output_file=None,
           predictions=None, truth=None,
           probabilities=None, probabilities_labels=None,
           target_is_encoded=False,
           error_message=None,
           models_count=None,
           training_duration=None,
           **others):
    return locals()


def output_subdir(name, config):
    subdir = os.path.join(config.output_dir, name, config.name, str(config.fold))
    touch(subdir, as_dir=True)
    return subdir


data_keys = re.compile("^(X|y|data)(_.+)?$")


def call_run(run_fn):
    import numpy as np

    params = NS.from_dict({"dataset":{"train":{"X_enc":"/tmp/dress/train.X_enc.npy","y_enc":"/tmp/dress/train.y_enc.npy"},
                                      "test":{"X_enc":"/tmp/dress/test.X_enc.npy","y_enc":"/tmp/dress/test.y_enc.npy"}},
                           "config":{"framewor k":"FEDOT","framework_params":{},"type":"classification","name":"Australian","fold":0,"metrics":["auc","logloss","acc"],"metric":"auc","seed":3029240368,"max_runtime_seconds":600,"cores":4,"max_mem_size_mb":91763,"min_vol_size_mb":-1,"input_dir":"/home/rosneft_user_2500/.openml/cache","output_dir":"/home/rosneft_user_2500/bench/automlbenchmark/results/fedot.small.test.local.20201225T163641","output_predictions_file":"/home/rosneft_user_2500/bench/automlbenchmark/results/fedot.small.test.local.20201225T163641/predictions/fedot.Australian.0.csv","result_token":"5e433616-46cf-11eb-a671-7957e32fc18d","result_dir":"/tmp/iris"}})

    def load_data(name, path, **ignored):
        if isinstance(path, str) and data_keys.match(name):
            return name, np.load(path, allow_pickle=True)
        return name, path

    print(params.dataset)
    ds = NS.walk(params.dataset, load_data)

    config = params.config
    config.framework_params = NS.dict(config.framework_params)

    try:
        result = run_fn(ds, config)
        res = dict(result)
        for name in ['predictions', 'truth', 'probabilities']:
            arr = result[name]
            if arr is not None:
                res[name] = os.path.join(config.result_dir, '.'.join([name, 'npy']))
                np.save(res[name], arr, allow_pickle=True)
    except Exception as e:
        log.exception(e)
        res = dict(
            error_message=str(e),
            models_count=0
        )

    print(config.result_token)
    print(json.dumps(res, separators=(',', ':')))


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info("\n**** FEDOT ****\n")

    is_classification = config.type == 'classification'
    # Mapping of benchmark metrics to FEDOT metrics
    metrics_mapping = dict(
        acc='accuracy',
        auc='roc_auc',
        f1='f1',
        logloss='neg_log_loss',
        mae='neg_mean_absolute_error',
        mse='neg_mean_squared_error',
        msle='neg_mean_squared_log_error',
        r2='r2',
        rmse='neg_mean_squared_error'
    )
    scoring_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if scoring_metric is None:
        raise ValueError("Performance metric {} not supported.".format(config.metric))

    if is_classification:
        metric = ClassificationMetricsEnum.ROCAUC
        task_type = TaskTypesEnum.classification
    else:
        metric = RegressionMetricsEnum.RMSE
        task_type = TaskTypesEnum.regression

    task = Task(task_type)

    x_train = dataset.train.X_enc
    y_train = dataset.train.y_enc

    x_test = dataset.test.X_enc

    x_train, y_train = shuffle(dataset.train.X_enc, dataset.train.y_enc, random_state = 0)

    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = np.squeeze(y_train, axis=1)

    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    dataset_to_compose = \
        InputData(idx=[_ for _ in range(len(y_train))],
                  features=x_train,
                  target=y_train,
                  task=task,
                  data_type=DataTypesEnum.table)

    dataset_to_test = \
        InputData(idx=[_ for _ in range(len(y_train))],
                  features=x_test,
                  target=None,
                  task=task,
                  data_type=DataTypesEnum.table)

    n_jobs = config.framework_params.get('_n_jobs',
                                         config.cores)  # useful to disable multicore, regardless of the dataset config

    log.info('Running FEDOT with a maximum time of %ss on %s cores, optimizing %s.',
             config.max_runtime_seconds, n_jobs, scoring_metric)
    runtime_min = (config.max_runtime_seconds / 60)

    available_model_types, _ = ModelTypesRepository().suitable_model(task_type=task.task_type)

    metric_function = MetricsRepository().metric_by_id(metric)

    # Create GP-based composer
    composer = GPComposer()

    if False:
        # the choice and initialisation of the GP search
        composer_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=3,
            max_depth=3, max_lead_time=datetime.timedelta(minutes=runtime_min * 0.8))

        # GP optimiser parameters choice
        scheme_type = GeneticSchemeTypesEnum.parameter_free
        optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

        # Create builder for composer and set composer params
        builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
            metric_function).with_optimiser_parameters(optimiser_parameters)

        composer = builder.build()

        # the optimal chain generation by composition - the most time-consuming task
        chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                    is_visualise=False)

    else:
        if is_classification:
            chain_evo_composed = Chain(PrimaryNode('logit'))
        else:
            chain_evo_composed = Chain(PrimaryNode('lasso'))

    chain_evo_composed.fit(input_data=dataset_to_compose, verbose=False)

    log.info('Predicting on the test set.')
    y_test = dataset.test.y_enc
    predictions = chain_evo_composed.predict(dataset_to_test, output_mode='labels').predict

    if not is_classification:
        probabilities = None
    else:
        probabilities = chain_evo_composed.predict(dataset_to_test, output_mode='full_probs').predict

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=1,
                  training_duration=1)


def save_artifacts(estimator, config):
    try:
        log.debug("All individuals :\n%s", list(estimator.evaluated_individuals_.items()))
        models = estimator.pareto_front_fitted_pipelines_
        hall_of_fame = list(zip(reversed(estimator._pareto_front.keys), estimator._pareto_front.items))
        artifacts = config.framework_params.get('_save_artifacts', False)
        if 'models' in artifacts:
            models_file = os.path.join(output_subdir('models', config), 'models.txt')
            with open(models_file, 'w') as f:
                for m in hall_of_fame:
                    pprint.pprint(dict(
                        fitness=str(m[0]),
                        model=str(m[1]),
                        pipeline=models[str(m[1])],
                    ), stream=f)
    except Exception:
        log.debug("Error when saving artifacts.", exc_info=True)


if __name__ == '__main__':
    call_run(run)
