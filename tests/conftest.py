import sys
from pathlib import Path

import pytest

import whydrift  # noqa: F401  (float64 before anything else)
from whydrift.physics.systems import make_system

sys.path.insert(0, str(Path(__file__).parent))  # so tests can `from fakes import ...`

from fakes import TrueFieldModel  # noqa: E402

# S3 parameters for tests only. The real FPUT form and values are [OPEN].
S3_TEST_KWARGS = dict(alpha=0.25, beta=0.0)


@pytest.fixture
def s1():
    return make_system("s1")


@pytest.fixture
def s2():
    return make_system("s2")


@pytest.fixture
def s3():
    return make_system("s3", **S3_TEST_KWARGS)


@pytest.fixture(params=["s1", "s2", "s3"])
def any_system(request):
    kwargs = S3_TEST_KWARGS if request.param == "s3" else {}
    return make_system(request.param, **kwargs)


@pytest.fixture
def fake_model(s1):
    return TrueFieldModel(s1)
