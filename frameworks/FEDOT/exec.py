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
from frameworks.shared.callee import call_run, result, output_subdir, utils
import numpy as np
import datetime

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

    with utils.Timer() as training:
        # the choice and initialisation of the GP search
        composer_requirements = GPComposerRequirements(
            primary=available_model_types,
            secondary=available_model_types, max_arity=3,
            max_depth=3, pop_size=3, num_of_generations=2,
            crossover_prob=0.8, mutation_prob=0.8, max_lead_time=datetime.timedelta(minutes=runtime_min))

        # GP optimiser parameters choice
        scheme_type = GeneticSchemeTypesEnum.steady_state
        optimiser_parameters = GPChainOptimiserParameters(genetic_scheme_type=scheme_type)

        # Create builder for composer and set composer params
        builder = GPComposerBuilder(task=task).with_requirements(composer_requirements).with_metrics(
            metric_function).with_optimiser_parameters(optimiser_parameters)

        composer = builder.build()

        # the optimal chain generation by composition - the most time-consuming task
        chain_evo_composed = composer.compose_chain(data=dataset_to_compose,
                                                    is_visualise=False)

        chain_evo_composed.fit(input_data=dataset_to_compose, verbose=False)

    log.info('Predicting on the test set.')
    y_test = dataset.test.y_enc
    predictions = chain_evo_composed.predict(dataset_to_test, output_mode='labels').predict

    if not is_classification:
        probabilities = None
    else:
        probabilities = chain_evo_composed.predict(dataset_to_test, output_mode='full_probs').predict

   # if is_classification:
        # if len(predictions.shape)==1
        #predictions = list(predictions)

        #if len(probabilities.shape) == 1:
        #    new_column = [1 - _ for _ in list(probabilities)]
        #    probabilities = np.expand_dims(probabilities, axis=1)
        #    new_column = np.expand_dims(new_column, axis=1)#
        #
        #    probabilities = np.append(probabilities, new_column, axis=1)
        #
        #    predictions = [0 if _ > np.mean(predictions) else 1 for _ in predictions]
        #else:
        #    predictions = [np.argmax(_) for _ in predictions]

            # print(probabilities.shape)

            # print(predictions.shape)

    # save_artifacts(fedot, config)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  truth=y_test,
                  probabilities=probabilities,
                  target_is_encoded=is_classification,
                  models_count=1,
                  training_duration=training.duration)


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
