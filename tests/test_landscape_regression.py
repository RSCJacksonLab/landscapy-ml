from pathlib import Path

import pytest

from landscapyml.landscape_regression import _read_split_indices


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_read_split_indices_rejects_unknown_labels_with_row_numbers(tmp_path):
    path = _write_csv(
        tmp_path / "unknown.csv",
        "sequence,set,validation\nAAA,train,false\nBBB,holdout,false\n",
    )

    with pytest.raises(ValueError, match=r"'holdout' \(rows 3\)"):
        _read_split_indices(
            path,
            split_column="set",
            validation_column="validation",
            train_label="train",
            test_label="test",
        )


def test_read_split_indices_rejects_blank_labels(tmp_path):
    path = _write_csv(
        tmp_path / "blank.csv",
        "sequence,set,validation\nAAA,train,false\nBBB,,false\n",
    )

    with pytest.raises(ValueError, match=r"'<blank>' \(rows 3\)"):
        _read_split_indices(
            path,
            split_column="set",
            validation_column="validation",
            train_label="train",
            test_label="test",
        )


def test_read_split_indices_normalizes_case_and_whitespace(tmp_path):
    path = _write_csv(
        tmp_path / "normalized.csv",
        'sequence,set,validation\nAAA," Train ",false\nBBB,TEST,false\n',
    )

    splits = _read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="TRAIN",
        test_label="test",
    )

    assert splits.train == [0]
    assert splits.val == []
    assert splits.test == [1]
    assert splits.counts() == {"train": 1, "val": 0, "test": 1, "total": 2}


def test_read_split_indices_gives_validation_precedence(tmp_path):
    path = _write_csv(
        tmp_path / "validation.csv",
        "sequence,set,validation\nAAA,unknown,true\nBBB,train,false\n",
    )

    splits = _read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="train",
        test_label="test",
    )

    assert splits.train == [1]
    assert splits.val == [0]
    assert splits.test == []


def test_read_split_indices_assigns_all_rows_when_split_column_is_absent(tmp_path):
    path = _write_csv(
        tmp_path / "no_split.csv",
        "sequence,validation\nAAA,false\nBBB,true\nCCC,false\n",
    )

    splits = _read_split_indices(
        path,
        split_column="set",
        validation_column="validation",
        train_label="train",
        test_label="test",
    )

    assert splits.train == [0, 2]
    assert splits.val == [1]
    assert splits.test == []
    assert splits.counts()["total"] == 3


def test_read_split_indices_rejects_empty_files(tmp_path):
    path = _write_csv(tmp_path / "empty.csv", "sequence,set,validation\n")

    with pytest.raises(ValueError, match=r"No training rows.*\(0 data rows\)"):
        _read_split_indices(
            path,
            split_column="set",
            validation_column="validation",
            train_label="train",
            test_label="test",
        )
