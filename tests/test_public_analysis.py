import copy
import csv
from pathlib import Path

import pytest

from scripts.analyze import check_tables, read_measurements, summarize
from src.common import write_csv


def test_public_accuracy_tables_match_measurements():
    rows = read_measurements('results/per_angle_accuracy.csv')
    check_tables(*summarize(rows), 'results')


def test_seed_pairing_known_accuracy_contrast():
    rows = read_measurements('results/per_angle_accuracy.csv')
    for row in rows:
        count = 78 + {42: 0, 123: 3, 456: 6}[int(row['seed'])]
        count += 6 if row['model'] == 'c4' else 0
        if row['regime'] == 'rotation':
            count += 3 if row['model'] == 'c4' else 12
        row['accuracy'] = str(count / 156)
    summary, comparisons = summarize(rows)
    baseline = next(r for r in summary if (r['model'], r['training_regime']) == ('cnn', 'original'))
    assert baseline['mean'] == pytest.approx(81 / 156)
    assert baseline['sd'] == pytest.approx(3 / 156)
    contrast = next(r for r in comparisons if r['comparison'] == 'c4/original minus cnn/original')
    assert contrast['mean'] == pytest.approx(6 / 156)
    assert contrast['sd'] == pytest.approx(0, abs=1e-15)


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'count', 'nonfinite', 'fraction', 'angle'])
def test_rejects_incomplete_or_invalid_measurements(tmp_path, defect):
    rows = read_measurements('results/per_angle_accuracy.csv')
    if defect == 'missing':
        rows.pop()
    elif defect == 'duplicate':
        rows[-1] = rows[0].copy()
    elif defect == 'count':
        rows[0]['n'] = '155'
    elif defect == 'nonfinite':
        rows[0]['accuracy'] = 'nan'
    elif defect == 'fraction':
        rows[0]['accuracy'] = '.5001'
    else:
        rows[0]['angle'] = '45'
    path = tmp_path / 'measurements.csv'
    write_csv(path, rows)
    with pytest.raises(ValueError):
        read_measurements(path)


def test_checks_reference_values_not_just_dimensions(tmp_path):
    summary, comparisons = summarize(read_measurements('results/per_angle_accuracy.csv'))
    wrong = copy.deepcopy(summary)
    wrong[0]['mean'] += .01
    write_csv(tmp_path / 'accuracy_summary.csv', wrong)
    write_csv(tmp_path / 'paired_accuracy.csv', comparisons)
    with pytest.raises(ValueError, match='value mismatch'):
        check_tables(summary, comparisons, tmp_path)


def test_analysis_refuses_existing_output(tmp_path, monkeypatch):
    from scripts.analyze import main
    monkeypatch.setattr('sys.argv', ['analyze', '--output', str(tmp_path)])
    with pytest.raises(FileExistsError):
        main()


def test_main_cli_routes_all_twelve_conditions_without_training(monkeypatch, tmp_path):
    import scripts.train as study
    calls = []
    monkeypatch.setattr(study, 'freeze', lambda *a: calls.append('freeze'))
    def fake_train(cfg, **kwargs):
        key = f"{cfg['model']}/{cfg['training_regime']}/{cfg['seed']}"
        calls.append(key)
        return tmp_path / key / 'best.pt'
    monkeypatch.setattr(study, 'train', fake_train)
    monkeypatch.setattr(study, 'evaluate', lambda p: calls.append(('evaluate', p)))
    monkeypatch.setattr(study, 'verify_run', lambda p, repeat=False: calls.append(('verify', p)))
    monkeypatch.setattr(study, 'archive_and_analyze', lambda *a: calls.append('analysis'))
    monkeypatch.setattr('sys.argv', ['train', '--config', 'configs/breastmnist.yaml', '--results', str(tmp_path)])
    study.main()
    assert calls[0] == 'freeze' and calls[-1] == 'analysis'
    assert len([c for c in calls if isinstance(c, str) and '/' in c]) == 12
    assert len([c for c in calls if isinstance(c, tuple) and c[0] == 'evaluate']) == 12
    assert len([c for c in calls if isinstance(c, tuple) and c[0] == 'verify']) == 12


@pytest.mark.parametrize('runner_name', ['scripts.train', 'scripts.train_matrix'])
def test_research_cli_rejects_validation_config_before_training(runner_name, monkeypatch, tmp_path):
    import importlib
    import yaml
    from src.common import settings
    cfg = settings('configs/breastmnist.yaml')
    cfg['track'] = 'validation'
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(cfg))
    runner = importlib.import_module(runner_name)
    def forbidden(*a, **k):
        pytest.fail('Validation config reached research training or test-data freeze')
    monkeypatch.setattr(runner, 'train', forbidden)
    if runner_name == 'scripts.train':
        monkeypatch.setattr(runner, 'freeze', forbidden)
    argv = [runner_name, '--config', str(config_path)]
    if runner_name == 'scripts.train':
        argv += ['--results', str(tmp_path / 'results')]
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
