import os
import logging
from yugiquery.utils.logging import LoggerConfig


def test_logger_basic():
    LoggerConfig.setup(level="DEBUG", log_file=None, logger_name="testlogger")
    LoggerConfig.propagate_env()
    logger = LoggerConfig.get_logger("testlogger")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    assert logger.level == logging.DEBUG
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert os.environ.get("YQ_LOG_LEVEL") == "DEBUG"
    LoggerConfig.clean_env()
    assert os.environ.get("YQ_LOG_LEVEL") is None


def test_logger_file(tmp_path):
    log_file = tmp_path / "log.txt"
    LoggerConfig.setup(level="INFO", log_file=str(log_file), logger_name="filelogger")
    logger = LoggerConfig.get_logger("filelogger")
    logger.info("File log message")
    LoggerConfig.propagate_env()
    assert os.environ.get("YQ_LOG_FILE") == str(log_file)
    logger.handlers[0].flush()
    with open(log_file) as f:
        content = f.read()
    assert "File log message" in content
    LoggerConfig.clean_env()
    assert os.environ.get("YQ_LOG_FILE") is None


def test_logger_env_propagation():
    def test_logger_multiple_setups(tmp_path):
        def test_logger_reconfigure_outside_loop(tmp_path):
            logger_name = "reconfiglogger"
            logger = LoggerConfig.get_logger(logger_name)
            configs = [
                {"level": "INFO", "log_file": None},
                {"level": "ERROR", "log_file": str(tmp_path / "reconfig1.txt")},
                {"level": "DEBUG", "log_file": None},
            ]
            for i, cfg in enumerate(configs):
                LoggerConfig.setup(level=cfg["level"], log_file=cfg["log_file"], logger_name=logger_name)
                logger = LoggerConfig.get_logger(logger_name)
                logger.info(f"Reconfig {i} info")
                logger.error(f"Reconfig {i} error")
                assert logger.level == logging._nameToLevel.get(cfg["level"].upper(), logging.INFO)
                if cfg["log_file"]:
                    logger.handlers[0].flush()
                    with open(cfg["log_file"]) as f:
                        content = f.read()
                    assert f"Reconfig {i} error" in content
                else:
                    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
                LoggerConfig.clean_env()

        configs = [
            {"level": "INFO", "log_file": None, "logger_name": "looplogger1"},
            {"level": "ERROR", "log_file": str(tmp_path / "loop1.txt"), "logger_name": "looplogger2"},
            {"level": "DEBUG", "log_file": None, "logger_name": "looplogger3"},
        ]
        for i, cfg in enumerate(configs):
            LoggerConfig.setup(**cfg)
            logger = LoggerConfig.get_logger(cfg["logger_name"])
            logger.info(f"Loop {i} info")
            logger.error(f"Loop {i} error")
            # Without env propagation
            assert os.environ.get("YQ_LOG_LEVEL") is None
            assert os.environ.get("YQ_LOG_FILE") is None
            # With env propagation
            LoggerConfig.propagate_env()
            assert os.environ.get("YQ_LOG_LEVEL") == cfg["level"]
            if cfg["log_file"]:
                assert os.environ.get("YQ_LOG_FILE") == cfg["log_file"]
            else:
                assert os.environ.get("YQ_LOG_FILE") is None
            LoggerConfig.clean_env()
            assert os.environ.get("YQ_LOG_LEVEL") is None
            assert os.environ.get("YQ_LOG_FILE") is None

    LoggerConfig.setup(level="WARNING", log_file=None, logger_name="envlogger")
    LoggerConfig.propagate_env()
    assert os.environ.get("YQ_LOG_LEVEL") == "WARNING"
    LoggerConfig.clean_env()
    assert os.environ.get("YQ_LOG_LEVEL") is None
