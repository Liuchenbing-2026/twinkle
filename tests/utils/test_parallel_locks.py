# Copyright (c) ModelScope Contributors. All rights reserved.
"""File coordination must not require a writable current working directory."""
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

from datasets.config import HF_DATASETS_CACHE


def load_parallel():
    source = Path(__file__).resolve().parents[2] / 'src/twinkle/utils/parallel.py'
    spec = importlib.util.spec_from_file_location('parallel_under_test', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_create_directory():
    with patch('os.makedirs', side_effect=PermissionError('read-only CWD')):
        load_parallel()


def test_processing_lock_uses_configured_cache(tmp_path, monkeypatch):
    module = load_parallel()
    monkeypatch.setattr(module, '_LOCK_DIR', str(tmp_path / 'cache/locks'))
    cwd = tmp_path / 'readonly'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    makedirs = os.makedirs

    def only_cache(path, *args, **kwargs):
        assert Path(path).is_absolute() and not Path(path).is_relative_to(cwd)
        return makedirs(path, *args, **kwargs)

    with patch('os.makedirs', side_effect=only_cache):
        with module.processing_lock('model/download'):
            pass
        with module.processing_lock('model/download'):
            pass
    assert not list(cwd.iterdir())
    assert len(list((tmp_path / 'cache/locks').glob('*.lock'))) == 1


def test_lock_location_respects_datasets_cache():
    module = load_parallel()
    assert Path(module._LOCK_DIR) == Path(HF_DATASETS_CACHE) / 'twinkle/locks'
