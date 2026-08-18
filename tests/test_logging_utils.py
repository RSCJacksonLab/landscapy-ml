import logging

import pytest

import landscapyml.logging_utils as logging_utils
from landscapyml.logging_utils import configure_logger


def _file_handlers(logger):
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.FileHandler)
    ]


def _console_handlers(logger):
    return [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
    ]


@pytest.fixture(autouse=True)
def isolated_package_logger():
    logger = logging.getLogger("landscapyml")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate
    logger.handlers = []
    yield
    for handler in logger.handlers:
        handler.close()
    logger.handlers = original_handlers
    logger.setLevel(original_level)
    logger.propagate = original_propagate


def test_configure_logger_file_only(tmp_path):
    log_path = tmp_path / "log.txt"

    logger = configure_logger(log_file=str(log_path), log_level="DEBUG")
    logger.debug("hello")

    assert len(_file_handlers(logger)) == 1
    assert _console_handlers(logger) == []
    assert "hello" in log_path.read_text()


def test_configure_logger_stream_only():
    logger = configure_logger(log_file=None, log_level="INFO")

    assert len(_console_handlers(logger)) == 1
    assert _file_handlers(logger) == []


def test_configure_logger_supports_combined_file_and_stream(tmp_path):
    logger = configure_logger(log_file=None)
    logger = configure_logger(log_file=str(tmp_path / "combined.log"))

    assert len(_console_handlers(logger)) == 1
    assert len(_file_handlers(logger)) == 1


def test_configure_logger_duplicate_calls_keep_handler_count_stable(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "stable.log"
    open_calls = []
    real_open = logging_utils._open_file_handler

    def tracking_open(path):
        open_calls.append(path)
        return real_open(path)

    monkeypatch.setattr(logging_utils, "_open_file_handler", tracking_open)
    logger = configure_logger(log_file=str(log_path))
    initial_count = len(logger.handlers)

    logger = configure_logger(log_file=str(log_path))

    assert len(logger.handlers) == initial_count
    assert open_calls == [log_path.resolve()]


def test_configure_logger_adds_distinct_file_paths(tmp_path):
    logger = configure_logger(log_file=str(tmp_path / "first.log"))
    logger = configure_logger(log_file=str(tmp_path / "second.log"))

    assert len(_file_handlers(logger)) == 2


def test_configure_logger_updates_levels_and_formatters_on_repeat(tmp_path):
    logger = configure_logger(log_file=None, log_level="DEBUG")
    logger = configure_logger(log_file=str(tmp_path / "level.log"), log_level="WARNING")

    assert logger.level == logging.WARNING
    assert all(handler.level == logging.WARNING for handler in logger.handlers)
    assert all(
        handler.formatter._fmt == "%(asctime)s %(levelname)s %(message)s"
        for handler in logger.handlers
    )


def test_configure_logger_disables_propagation():
    logger = configure_logger()

    assert logger.propagate is False
