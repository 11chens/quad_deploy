from loguru import logger
import time
import sys
from typing import Set, Optional


class CustomLogger:
    """A custom logger class based on Loguru with once, throttle, and colored terminal output."""

    def __init__(
        self, name: str = __name__, log_file: Optional[str] = None, rotation: str = "10 MB",
        format: str = "<green>{time:YY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <white>{message}</white>"
    ):
        """
        Initialize the custom logger with colored terminal output and optional file output.
        
        Args:
            name: Logger name, defaults to module name.
            log_file: Path to log file, if None, logs to terminal only.
            rotation: Log file rotation setting (e.g., '10 MB' or '1 day').
            format: Log message format with color tags for terminal.
        """
        self.logger = logger.bind(name=name)
        self.printed_messages: Set[str] = set()
        self.throttle_times = {}

        # Remove default sink to avoid duplicate output
        self.logger.remove()

        # Add terminal sink with colored output
        self.logger.add(
            sink=sys.stderr,
            level="DEBUG",
            format=format,
            colorize=True  # Explicitly enable colored output
        )

        # Add file sink if specified (without color tags)
        if log_file:
            self.logger.add(
                sink=log_file,
                rotation=rotation,
                level="DEBUG",
                format=format.replace("<green>", "").replace("</green>", "").replace("<level>", "").replace(
                    "</level>", "").replace("<white>", "").replace("</white>", ""),  # Remove color tags for file
                colorize=False)

    def log_once(self, message: str, level: str):
        """
        Log a message only once.
        
        Args:
            message: The message to log.
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        if message not in self.printed_messages:
            getattr(self.logger, level.lower())(message)
            self.printed_messages.add(message)

    def log_throttle(self, message: str, seconds: float, level: str = "INFO"):
        """
        Log a message with a minimum time interval between logs.
        
        Args:
            message: The message to log.
            seconds: Minimum interval between logs in seconds.
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        current_time = time.time()
        last_log_time = self.throttle_times.get(message, 0)
        if current_time - last_log_time >= seconds:
            getattr(self.logger, level.lower())(message)
            self.throttle_times[message] = current_time

    def _log(self, message: str, level: str = "INFO"):
        """
        Log a message without restrictions.
        
        Args:
            message: The message to log.
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        getattr(self.logger, level.lower())(message)

    def debug(self, message: str, once=False):
        if once:
            self.log_once(message, "debug")
        else:
            getattr(self.logger, "debug")(message)

    def info(self, message: str, once=False):
        if once:
            self.log_once(message, "info")
        else:
            getattr(self.logger, "info")(message)

    def warning(self, message: str, once=False):
        if once:
            self.log_once(message, "warning")
        else:
            getattr(self.logger, "warning")(message)

    def error(self, message: str, once=False):
        if once:
            self.log_once(message, "error")
        else:
            getattr(self.logger, "error")(message)

    def critical(self, message: str, once=False):
        if once:
            self.log_once(message, "critical")
        else:
            getattr(self.logger, "critical")(message)


# Example usage
if __name__ == "__main__":
    # Initialize logger with terminal and file output
    custom_logger = CustomLogger(name="my_app", log_file="app.log")

    # Test log_once
    for _ in range(3):
        custom_logger.log_once("This will only print once", level="INFO")
        time.sleep(1)

    # Test log_throttle
    for _ in range(6):
        custom_logger.log_throttle("This will print every 2 seconds", seconds=2, level="WARNING")
        time.sleep(1)

    # Test regular log
    custom_logger.debug("This is debug")
    custom_logger.info("This is info")
    custom_logger.warning("This is warning")
    custom_logger.error("This is error")
    custom_logger.critical("This is critical")
