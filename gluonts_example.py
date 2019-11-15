"""
pip install gluonts==0.3.3
pip install git+https://github.com/awslabs/gluon-ts.git  # needed for deep state and deep factor

Note the output data in this example is a bit weird in my opinion:
For the Target data, each row is a time series, with the columns being the datetimes

        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min

target shape: (num_series, ts_length)
feat_dynamic_real shape: (num_series, num_features, ts_length)

Written using 0.3.3, but then changed to the github version which has some new models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from gluonts.dataset.common import ListDataset
from gluonts.dataset.util import to_pandas

try:
    from gluonts.transform import FieldName # old way (0.3.3 and older)
except:
    from gluonts.dataset.field_names import FieldName # new way

def create_dataset(num_series, num_steps, period=24, mu=1, sigma=0.3):
    # create target: noise + pattern    
    # noise
    noise = np.random.normal(mu, sigma, size=(num_series, num_steps))
    
    # pattern - sinusoid with different phase
    sin_minumPi_Pi = np.sin(np.tile(np.linspace(-np.pi, np.pi, period), int(num_steps / period)))
    sin_Zero_2Pi = np.sin(np.tile(np.linspace(0, 2 * np.pi, 24), int(num_steps / period)))
    
    pattern = np.concatenate((np.tile(sin_minumPi_Pi.reshape(1, -1), 
                                      (int(np.ceil(num_series / 2)),1)), 
                              np.tile(sin_Zero_2Pi.reshape(1, -1), 
                                      (int(np.floor(num_series / 2)), 1))
                             ),
                             axis=0
                            )
    
    target = noise + pattern
    
    # create time features: use target one period earlier, append with zeros
    feat_dynamic_real = np.concatenate((np.zeros((num_series, period)), 
                                        target[:, :-period]
                                       ), 
                                       axis=1
                                      )
    
    # create categorical static feats: use the sinusoid type as a categorical feature
    feat_static_cat = np.concatenate((np.zeros(int(np.ceil(num_series / 2))), 
                                      np.ones(int(np.floor(num_series / 2)))
                                     ),
                                     axis=0
                                    )
    
    return target, feat_dynamic_real, feat_static_cat

# define the parameters of the dataset
custom_ds_metadata = {'num_series': 100,
                      'num_steps': 24 * 7,
                      'prediction_length': 24,
                      'context_length': 24 * 2,
                      'freq': '1H',
                      'start': [pd.Timestamp("01-01-2019", freq='1H') 
                                for _ in range(100)]
                     }
data_out = create_dataset(custom_ds_metadata['num_series'], 
                          custom_ds_metadata['num_steps'],                                                      
                          custom_ds_metadata['prediction_length']
                         )

target, feat_dynamic_real, feat_static_cat = data_out

train_ds = ListDataset([{FieldName.TARGET: target, 
                         FieldName.START: start,
                         FieldName.FEAT_DYNAMIC_REAL: fdr,
                         FieldName.FEAT_STATIC_CAT: fsc} 
                        for (target, start, fdr, fsc) in zip(target[:, :-custom_ds_metadata['prediction_length']], 
                                                             custom_ds_metadata['start'], 
                                                             feat_dynamic_real[:, :-custom_ds_metadata['prediction_length']], 
                                                             feat_static_cat)],
                      freq=custom_ds_metadata['freq'])
test_ds = ListDataset([{FieldName.TARGET: target, 
                        FieldName.START: start,
                        FieldName.FEAT_DYNAMIC_REAL: fdr,
                        FieldName.FEAT_STATIC_CAT: fsc} 
                       for (target, start, fdr, fsc) in zip(target, 
                                                            custom_ds_metadata['start'], 
                                                            feat_dynamic_real, 
                                                            feat_static_cat)],
                     freq=custom_ds_metadata['freq'])
                       
train_entry = next(iter(train_ds))
train_entry.keys()

test_entry = next(iter(test_ds))
test_entry.keys()

test_series = to_pandas(test_entry)
train_series = to_pandas(train_entry)

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 7))

train_series.plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].legend(["train series"], loc="upper left")

test_series.plot(ax=ax[1])
ax[1].axvline(train_series.index[-1], color='r') # end of train dataset
ax[1].grid(which="both")
ax[1].legend(["test series", "end of train series"], loc="upper left")

plt.show()

##########################
"""
TRANSFORM
"""

from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    SetFieldIfNotPresent,
)

def create_transformation(freq, context_length, prediction_length):
    return Chain(
        [
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=prediction_length,
                log_scale=True,
            ),
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                past_length=context_length,
                future_length=prediction_length,
                time_series_fields=[
                    FieldName.FEAT_AGE,
                    FieldName.FEAT_DYNAMIC_REAL,
                    FieldName.OBSERVED_VALUES,
                ],
            ),
        ]
    )

transformation = create_transformation(custom_ds_metadata['freq'], 
                                       2 * custom_ds_metadata['prediction_length'], # can be any appropriate value
                                       custom_ds_metadata['prediction_length'])

train_tf = transformation(iter(train_ds), is_train=True)


"""
MODEL TRAINING

https://docs.aws.amazon.com/forecast/latest/dg/aws-forecast-recipe-npts.html

'We can create a simple training network that defines a neural network that 
takes as input a window of length context_length and predicts the 
subsequent window of dimension prediction_length 
(thus, the output dimension of the network is prediction_length)'
"""
from gluonts.trainer import Trainer
"""
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=custom_ds_metadata['prediction_length'],
    context_length=custom_ds_metadata['context_length'],
    freq=custom_ds_metadata['freq'],
    trainer=Trainer(ctx="cpu", 
                    epochs=10, 
                    learning_rate=1e-3, 
                    hybridize=False, 
                    num_batches_per_epoch=100
                   )
)

from gluonts.model.deepar import DeepAREstimator
estimator = DeepAREstimator(freq=custom_ds_metadata['freq'],
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'], 
                            trainer=Trainer(epochs=100))

from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
estimator = DeepAREstimator(freq=custom_ds_metadata['freq'],
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'], 
                            trainer=Trainer(epochs=100),
                            distr_output=PiecewiseLinearOutput(8))

from gluonts.model.deepar import DeepAREstimator
estimator = DeepAREstimator(freq=custom_ds_metadata['freq'],
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'], 
                            trainer=Trainer(epochs=10),
                            use_feat_static_cat = True,
                            cardinality = [2])


from gluonts.model.seq2seq import MQCNNEstimator
estimator = MQCNNEstimator(freq=custom_ds_metadata['freq'], 
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'], 
                            trainer=Trainer(epochs=10))

from gluonts.model.wavenet import WaveNetEstimator
estimator = WaveNetEstimator(freq=custom_ds_metadata['freq'], 
                                   prediction_length=custom_ds_metadata['prediction_length'],
                                   trainer=Trainer(epochs=20))

from gluonts.model.deep_factor import DeepFactorEstimator # not in by version 0.3.3
estimator = DeepFactorEstimator(freq=custom_ds_metadata['freq'], 
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'], 
                            trainer=Trainer(epochs=10))

from gluonts.model.gp_forecaster import GaussianProcessEstimator
estimator = GaussianProcessEstimator(freq=custom_ds_metadata['freq'],
                                     cardinality=custom_ds_metadata['num_series'],
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'], 
                            trainer=Trainer(epochs=10))

from gluonts.model.npts import NPTSEstimator
estimator = NPTSEstimator(freq=custom_ds_metadata['freq'],
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'])

from gluonts.model.canonical import CanonicalRNNEstimator  
estimator = CanonicalRNNEstimator(freq=custom_ds_metadata['freq'],
                            context_length=custom_ds_metadata['context_length'],
                            prediction_length=custom_ds_metadata['prediction_length'], 
                            # use_feat_dynamic_real = True,
                            trainer=Trainer(epochs=20))

# a bit different, slow
from gluonts.model.prophet import ProphetPredictor  
estimator = ProphetPredictor(freq=custom_ds_metadata['freq'],
                            prediction_length=custom_ds_metadata['prediction_length'])
forecasts = list(estimator.predict(train_ds))
forecasts[0].samples
forecasts[0].quantile(0.5)
"""

from gluonts.model.seasonal_naive import SeasonalNaiveEstimator
estimator = SeasonalNaiveEstimator(freq=custom_ds_metadata['freq'], 
                            prediction_length=custom_ds_metadata['prediction_length'])

predictor = estimator.train(train_ds)


"""
EVALUATION
"""

from gluonts.evaluation.backtest import make_evaluation_predictions

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_eval_samples=100,  # number of sample paths we want for evaluation
)
forecasts = list(forecast_it)
tss = list(ts_it)


from gluonts.evaluation import Evaluator

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_ds))

print(json.dumps(agg_metrics, indent=4))
# which metrics? https://robjhyndman.com/papers/foresight.pdf

item_metrics.plot(x='MSIS', y='MASE', kind='scatter')
plt.grid(which="both")
plt.show()

def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()

ts_entry = tss[0]
forecast_entry = forecasts[0]
plot_prob_forecasts(ts_entry, forecast_entry)


# FeedForward 5 epoch sMAPE 0.5427, RMSE: 0.328, MSIS: 6.746909
# FeedForward 50 epoch sMAPE 0.54605, RMSE: 0.32359, MSIS: 6.4502
# DeepAR (1/2 context) sMAPE 0.536, RMSE: 0.35357, MSIS: 8.90759
# DeepAR (full context) sMAPE 0.506875, RMSE: 0.33369, MSIS: 8.07169
# DeepAR (full 100 ep) sMAPE 0.523305, RMSE: 0.33396, MSIS: 6.74545
# DeepAR (full 100 ep, PLO) sMAPE 0.507, RMSE: 0.349, MSIS: 7.54163
# DeepAR (full 10 ep, cat vars) sMAPE 0.5342, RMSE: 0.37887, MSIS: 9.3577
# MQCNN (10 ep) sMAPE 0.5322, RMSE: NaN, MSIS: 6.346
# Seasonal Naive sMAPE 0.6199, RMSE: 0.4201, MSIS: 40.1074
# Gausian Process: sMAPE: 1.802, RMSE: 1.26, MSIS: 54.334
# NPTS, sMAPE: 0.649, RMSE: 0.47186,MSIS: 44.033