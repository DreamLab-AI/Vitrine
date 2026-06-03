# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for UTF-8 path handling in Python bindings."""

import base64
from pathlib import Path
import time

import pytest


@pytest.mark.integration
def test_import_related_bindings_accept_unicode_paths(lf, tmp_path):
    dataset_path = tmp_path / "日本語_データセット"
    output_path = tmp_path / "出力フォルダ"
    init_path = tmp_path / "初期化ポイント.ply"
    checkpoint_path = tmp_path / "チェックポイント.resume"
    config_path = tmp_path / "設定.json"
    config_save_path = tmp_path / "保存設定.json"
    keymap_path = tmp_path / "操作設定.json"
    image_path = tmp_path / "画像.png"

    dataset_path.mkdir()
    output_path.mkdir()
    config_path.write_text("{}", encoding="utf-8")
    image_path.write_bytes(
        base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4////fwAJ+wP9KobjigAAAABJRU5ErkJggg==")
    )

    dataset_info = lf.detect_dataset_info(str(dataset_path))
    assert dataset_info is not None
    assert str(dataset_info.base_path) == str(dataset_path)

    lf.load_file(
        str(dataset_path),
        is_dataset=True,
        output_path=str(output_path),
        init_path=str(init_path),
    )
    lf.load_checkpoint_for_training(
        str(checkpoint_path),
        str(dataset_path),
        str(output_path),
    )
    lf.load_config_file(str(config_path))
    lf.app.open(str(dataset_path))

    with pytest.raises(RuntimeError, match="No parameter manager available"):
        lf.save_config_file(str(config_save_path))

    assert lf.keymap.export_profile(str(keymap_path)) is False
    assert lf.keymap.import_profile(str(keymap_path)) is False

    width, height, channels = lf.ui.get_image_info(str(image_path))
    assert (width, height) == (1, 1)
    assert channels in (3, 4)

    lf.ui.preload_image_async(str(image_path))
    for _ in range(100):
        if lf.ui.is_preload_ready(str(image_path)):
            break
        time.sleep(0.01)
    assert lf.ui.is_preload_ready(str(image_path)) is True
    lf.ui.cancel_preload(str(image_path))
