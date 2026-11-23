import jax

from fakes import ExactStepModel, TrueFieldModel

from whydrift.common import MODELS, SEPARABLE_MODELS
from whydrift.contracts import METRIC_COLUMNS, Model, is_separable


def test_float64_is_on():
    assert jax.config.jax_enable_x64


def test_fakes_satisfy_model_contract(s1):
    assert isinstance(TrueFieldModel(s1), Model)
    assert isinstance(ExactStepModel(0.1), Model)


def test_is_separable(s1):
    assert is_separable(TrueFieldModel(s1))
    assert not is_separable(ExactStepModel(0.1))


def test_metric_columns_unique():
    assert len(METRIC_COLUMNS) == len(set(METRIC_COLUMNS))
    assert {"b", "s", "b_abs", "b_val", "drift_t1000", "vpt"} <= set(METRIC_COLUMNS)


def test_separable_models_are_known():
    assert set(SEPARABLE_MODELS) <= set(MODELS)
