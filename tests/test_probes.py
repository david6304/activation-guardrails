"""Tests for agguardrails.probes."""

import numpy as np

from agguardrails.probes import fit_probe_for_layer, select_best_probe


def test_fit_probe_for_layer_returns_metrics():
    x_train = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [1.1]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_val = np.array([[0.05], [0.15], [0.95], [1.05]])
    y_val = np.array([0, 0, 1, 1])
    x_test = np.array([[0.03], [0.12], [0.92], [1.02]])
    y_test = np.array([0, 0, 1, 1])

    result = fit_probe_for_layer(
        layer=8,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        c=1.0,
        max_iter=1000,
        target_fpr=0.0,
        random_state=42,
    )

    assert result.layer == 8
    assert result.val_result.roc_auc >= 0.5
    assert result.test_result.roc_auc >= 0.5


def test_select_best_probe_prefers_validation_tpr():
    x_train = np.array([[0.0], [0.1], [0.2], [0.9], [1.0], [1.1]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    x_val = np.array([[0.05], [0.15], [0.95], [1.05]])
    y_val = np.array([0, 0, 1, 1])
    x_test = np.array([[0.03], [0.12], [0.92], [1.02]])
    y_test = np.array([0, 0, 1, 1])

    weak = fit_probe_for_layer(
        layer=8,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        c=0.01,
        max_iter=1000,
        target_fpr=0.0,
        random_state=42,
    )
    strong = fit_probe_for_layer(
        layer=16,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        c=1.0,
        max_iter=1000,
        target_fpr=0.0,
        random_state=42,
    )

    best = select_best_probe([weak, strong])
    assert best.layer in {8, 16}
