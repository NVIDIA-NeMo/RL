import pytest
from logra.config import LoGRAConfig
from pydantic import ValidationError


def test_unsupported_optimizer_and_bad_rank_fail():
    with pytest.raises(ValidationError):
        LoGRAConfig(optimizer="adam")
    with pytest.raises(ValidationError):
        LoGRAConfig(rank=0)


def test_predicted_kl_is_not_a_project_option():
    with pytest.raises(ValidationError):
        LoGRAConfig(kl_budget=0.01)
