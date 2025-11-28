import logging
from pathlib import Path

from landscapyml.logging_utils import configure_logger


def test_configure_logger_writes_file_and_stream(tmp_path):
    log_path = tmp_path / "log.txt"
    logger = configure_logger(log_file=str(log_path), log_level="DEBUG")
    logger.debug("hello")
    assert log_path.exists()
    assert "hello" in log_path.read_text()

    # Stream handler should be attached when no log_file is provided
    stream_logger = configure_logger(log_file=None, log_level="INFO")
    assert any(isinstance(h, logging.StreamHandler) for h in stream_logger.handlers)
    stream_logger.info("stream")
