import pytest

import astrospec
from astrospec.registry import create_model, is_model, list_models, register_model


def test_known_models_are_registered():
    assert is_model("galspecnet")


def test_list_models_is_sorted():
    models = list_models()
    assert models == sorted(models)


def test_create_model_unknown_name_raises():
    assert not is_model("not_a_real_model")
    with pytest.raises(ValueError, match="not_a_real_model"):
        create_model("not_a_real_model")


def test_register_model_adds_to_registry():
    @register_model
    def _dummy_registry_test_model(**kwargs):
        return kwargs

    assert is_model("_dummy_registry_test_model")
    assert create_model("_dummy_registry_test_model", x=1) == {"x": 1}


def test_astrospec_reexports_registry_api():
    assert astrospec.create_model is create_model
    assert astrospec.is_model is is_model
    assert astrospec.list_models is list_models
    assert astrospec.register_model is register_model
