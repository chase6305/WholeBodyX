"""Comparison reports must use matching experiments and unsampled force data."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def report_module():
    path = Path(__file__).resolve().parents[2] / 'scripts/report_h1_dynamics.py'
    spec = importlib.util.spec_from_file_location('dynamics_report', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_trace(path, *, change=None, count=3):
    rows = [dict(time=.005 * (i + 1), target_com_y_m=0., com_y_m=.001,
                 com_error_m=.001, left_fz_n=300. + [0., 2., -1.][i], right_fz_n=300.,
                 contact_position_error_m=1e-6, applied_push_y_n=0.) for i in range(count)]
    if change is not None:
        key, value = change
        rows[-1][key] = value
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_comparison_reports_raw_force_steps_and_sample_intervals(report_module, tmp_path):
    a, b = write_trace(tmp_path/'a.csv'), write_trace(tmp_path/'b.csv')
    _, summary = report_module.compare(a, b)
    for data in summary.values():
        assert data['com_rmse_mm'] == pytest.approx(1.)
        assert data['max_foot_drift_um'] == pytest.approx(1.)
        assert data['max_force_step_n'] == 3.
        assert data['force_step_rms_n'] == pytest.approx(np.sqrt(13./4.))
        assert data['sample_interval_min_s'] == pytest.approx(.005)
        assert data['sample_interval_max_s'] == pytest.approx(.005)


@pytest.mark.parametrize('change', [('time', .016), ('target_com_y_m', .1), ('applied_push_y_n', 3.)])
def test_report_rejects_different_experiments_before_writing(report_module, tmp_path, change):
    a = write_trace(tmp_path/'a.csv')
    b = write_trace(tmp_path/'b.csv', change=change)
    output = tmp_path/'report.html'
    with pytest.raises(ValueError, match='matching ' + change[0]):
        report_module.report(a, b, output)
    assert not output.exists()
    assert not output.with_suffix('.json').exists()


def test_comparison_rejects_different_lengths(report_module, tmp_path):
    with pytest.raises(ValueError, match='matching time'):
        report_module.compare(write_trace(tmp_path/'a.csv'), write_trace(tmp_path/'b.csv', count=2))


@pytest.mark.parametrize('change', [('time', .005), ('com_error_m', float('nan'))])
def test_report_rejects_invalid_samples(report_module, tmp_path, change):
    with pytest.raises(ValueError, match='finite samples and increasing times'):
        report_module.load(write_trace(tmp_path/'a.csv', change=change))


def test_report_needs_two_samples_for_force_steps(report_module, tmp_path):
    with pytest.raises(ValueError, match='at least two'):
        report_module.load(write_trace(tmp_path/'a.csv', count=1))
