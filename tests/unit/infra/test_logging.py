# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for unified logging interface."""

import logging
import tempfile
from pathlib import Path

import pytest

from nemo_rl.infra.logging import (
    LoggerFacade,
    LogLevel,
    configure_logging,
    get_logger,
    log_debug,
    log_error,
    log_info,
    log_warning,
)


class TestLogLevel:
    """Tests for LogLevel enum."""

    def test_to_logging_level_debug(self):
        """Test DEBUG level conversion."""
        assert LogLevel.DEBUG.to_logging_level() == logging.DEBUG

    def test_to_logging_level_info(self):
        """Test INFO level conversion."""
        assert LogLevel.INFO.to_logging_level() == logging.INFO

    def test_to_logging_level_warning(self):
        """Test WARNING level conversion."""
        assert LogLevel.WARNING.to_logging_level() == logging.WARNING

    def test_to_logging_level_error(self):
        """Test ERROR level conversion."""
        assert LogLevel.ERROR.to_logging_level() == logging.ERROR

    def test_to_logging_level_critical(self):
        """Test CRITICAL level conversion."""
        assert LogLevel.CRITICAL.to_logging_level() == logging.CRITICAL


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_default_settings(self):
        """Test configuration with default settings."""
        configure_logging()
        logger = get_logger("test_default")
        assert logger is not None

    def test_configure_with_level(self):
        """Test configuration with custom level."""
        configure_logging(level=LogLevel.DEBUG)
        logger = get_logger("test_level")
        assert logger.level == logging.DEBUG

    def test_configure_with_string_level(self):
        """Test configuration with string level."""
        configure_logging(level="warning")
        logger = get_logger("test_str_level")
        # Root logger should be set to WARNING
        assert logging.getLogger().level == logging.WARNING

    def test_configure_with_log_file(self):
        """Test configuration with log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            configure_logging(log_file=log_file)

            logger = get_logger("test_file")
            logger.info("Test message")

            # File should be created
            assert log_file.exists()


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_named_logger(self):
        """Test getting a named logger."""
        logger = get_logger("my_module")
        assert logger is not None
        assert logger.name == "my_module"

    def test_get_same_logger_twice(self):
        """Test getting the same logger returns same instance."""
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        assert logger1 is logger2

    def test_get_root_logger(self):
        """Test getting root logger with None."""
        logger = get_logger(None)
        assert logger.name == "root"


class TestLoggerFacade:
    """Tests for LoggerFacade class."""

    def test_create_facade(self):
        """Test creating a logger facade."""
        facade = LoggerFacade("test_facade")
        assert facade.name == "test_facade"
        assert facade.logger is not None

    def test_info_logging(self, caplog):
        """Test info level logging."""
        facade = LoggerFacade("test_info")
        with caplog.at_level(logging.INFO):
            facade.info("Test info message")
        assert "Test info message" in caplog.text

    def test_debug_logging(self, caplog):
        """Test debug level logging."""
        facade = LoggerFacade("test_debug")
        with caplog.at_level(logging.DEBUG):
            facade.debug("Test debug message")
        assert "Test debug message" in caplog.text

    def test_warning_logging(self, caplog):
        """Test warning level logging."""
        facade = LoggerFacade("test_warning")
        with caplog.at_level(logging.WARNING):
            facade.warning("Test warning message")
        assert "Test warning message" in caplog.text

    def test_error_logging(self, caplog):
        """Test error level logging."""
        facade = LoggerFacade("test_error")
        with caplog.at_level(logging.ERROR):
            facade.error("Test error message")
        assert "Test error message" in caplog.text

    def test_critical_logging(self, caplog):
        """Test critical level logging."""
        facade = LoggerFacade("test_critical")
        with caplog.at_level(logging.CRITICAL):
            facade.critical("Test critical message")
        assert "Test critical message" in caplog.text

    def test_structured_logging(self, caplog):
        """Test structured logging."""
        facade = LoggerFacade("test_structured")
        with caplog.at_level(logging.INFO):
            facade.log_structured(
                "training_step",
                level="info",
                step=100,
                loss=0.5,
                reward=1.2,
            )
        assert "training_step" in caplog.text

    def test_set_level(self):
        """Test setting log level."""
        facade = LoggerFacade("test_set_level")
        facade.set_level(LogLevel.DEBUG)
        assert facade.logger.level == logging.DEBUG

        facade.set_level("warning")
        assert facade.logger.level == logging.WARNING


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    def test_log_info(self, caplog):
        """Test log_info function."""
        with caplog.at_level(logging.INFO, logger="nemo_rl"):
            log_info("Info message from module")
        assert "Info message from module" in caplog.text

    def test_log_debug(self, caplog):
        """Test log_debug function."""
        with caplog.at_level(logging.DEBUG, logger="nemo_rl"):
            log_debug("Debug message from module")
        assert "Debug message from module" in caplog.text

    def test_log_warning(self, caplog):
        """Test log_warning function."""
        with caplog.at_level(logging.WARNING, logger="nemo_rl"):
            log_warning("Warning message from module")
        assert "Warning message from module" in caplog.text

    def test_log_error(self, caplog):
        """Test log_error function."""
        with caplog.at_level(logging.ERROR, logger="nemo_rl"):
            log_error("Error message from module")
        assert "Error message from module" in caplog.text


class TestLoggingConsistency:
    """Tests for logging consistency across modules."""

    def test_same_format_different_modules(self, caplog):
        """Test that different modules use consistent formatting."""
        facade1 = LoggerFacade("module_a")
        facade2 = LoggerFacade("module_b")
        facade3 = LoggerFacade("module_c")

        with caplog.at_level(logging.INFO):
            facade1.info("Message from A")
            facade2.info("Message from B")
            facade3.info("Message from C")

        # All messages should be in the log
        assert "Message from A" in caplog.text
        assert "Message from B" in caplog.text
        assert "Message from C" in caplog.text
