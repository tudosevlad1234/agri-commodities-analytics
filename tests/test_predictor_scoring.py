import math

import numpy as np
import pandas as pd

from agri import config as C
from agri import predictor as pr
from agri import scoring as sc


def test_walk_forward_has_out_of_sample_metrics(corn_report):
    result = corn_report.predictor
    assert result.ok
    assert result.n_oos > 500
    assert 0.4 < result.auc < 0.8
    assert 0.0 < result.p_up < 1.0
    assert abs(sum(result.importances.values()) - 1.0) < 1e-9


def test_predictions_do_not_depend_on_unseen_labels(corn_report):
    frame = corn_report.features
    base = pr.walk_forward_predict(frame)
    cut_row = len(frame) - 200
    altered = frame.copy()
    altered.iloc[cut_row:, altered.columns.get_loc("target")] = 5.0
    again = pr.walk_forward_predict(altered)
    boundary = frame.index[cut_row - C.WALK_FORWARD_REFIT_WEEKS - C.HORIZON_WEEKS]
    shared = base.predictions.index[base.predictions.index < boundary]
    assert np.allclose(base.predictions.loc[shared, "p_up"], again.predictions.loc[shared, "p_up"])


def test_base_rate_only_uses_known_labels():
    index = pd.date_range("2000-01-07", periods=200, freq="W-FRI")
    target = pd.Series(np.where(np.arange(200) < 100, 0.01, -0.01), index=index)
    rate = pr.base_rate(target, 4)
    assert rate.iloc[: 52 + 3].isna().all()
    assert rate.iloc[100 + 3] == 1.0
    assert rate.iloc[100 + 4] < 1.0


def test_recency_weights_decay():
    index = pd.date_range("2000-01-01", periods=5, freq="365D")
    weights = pr.recency_weights(index)
    assert weights[-1] == 1.0 and np.all(np.diff(weights) > 0)


def test_short_history_is_rejected(corn_report):
    assert not pr.walk_forward_predict(corn_report.features.iloc[:300]).ok


def test_family_weights_sum_to_one(corn_report):
    ic = sc.rolling_ic(corn_report.features, C.HORIZON_WEEKS)
    weights = sc.family_weights(ic).dropna()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights >= 0).all().all()


def test_verdict_thresholds():
    assert sc.verdict_for(50) == "BULLISH (strong)"
    assert sc.verdict_for(-20) == "Bearish"
    assert sc.verdict_for(3) == "Neutral"
    assert sc.verdict_for(float("nan")) == "n/a"


def test_ml_weight_requires_skill():
    assert sc.ml_weight_from_skill(None) == 0.0
    weak = pr.PredictorResult(True, auc=0.49, brier_skill=0.02)
    miscalibrated = pr.PredictorResult(True, auc=0.7, brier_skill=-0.01)
    strong = pr.PredictorResult(True, auc=0.7, brier_skill=0.02)
    assert sc.ml_weight_from_skill(weak) == 0.0
    assert sc.ml_weight_from_skill(miscalibrated) == 0.0
    assert abs(sc.ml_weight_from_skill(strong) - sc.ML_MAX_WEIGHT) < 1e-9


def test_probability_falls_back_to_base_rate_without_skill(corn_report):
    features = corn_report.features
    unskilled = pr.PredictorResult(True, auc=0.6, brier_skill=-0.05, p_up=0.2, base_rate=0.52)
    result = sc.score_contract(features, corn_report.fair_value, corn_report.regime, unskilled)
    assert result.p_up == 0.52 and result.ml_weight == 0.0
    skilled = pr.PredictorResult(True, auc=0.6, brier_skill=0.05, p_up=0.2, base_rate=0.52)
    result = sc.score_contract(features, corn_report.fair_value, corn_report.regime, skilled)
    assert result.p_up == 0.2 and abs(result.ml_weight - sc.ML_MAX_WEIGHT) < 1e-9


def test_agreement_counts_only_active_signals():
    signals = {"a": 0.8, "b": -0.8, "c": 0.01}
    weights = {"a": 0.5, "b": 0.25, "c": 0.25}
    assert math.isclose(sc.agreement_part(signals, weights, 0.3), 0.5 / 0.75)


def test_trust_is_capped_without_skill(corn_report):
    features = corn_report.features
    result = sc.score_contract(features, corn_report.fair_value, corn_report.regime, None)
    assert result.trust_parts["model_skill"] == 0.0
    assert result.trust <= C.TRUST_CAP_NO_SKILL
    assert result.ml_weight == 0.0
    assert abs(result.score - 100 * result.composite) < 1e-9


def test_trust_weights_sum_to_one():
    assert abs(sum(C.TRUST_WEIGHTS.values()) - 1.0) < 1e-12


def test_score_is_bounded(corn_report):
    assert -100 <= corn_report.score.score <= 100
    assert 0 <= corn_report.score.trust <= 100
