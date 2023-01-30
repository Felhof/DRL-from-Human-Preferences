import logging
import logging.handlers
import os
from multiprocessing import Process, Queue
from time import sleep
from typing import Dict

PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LOG_DIRECTORY = "logs"
LOG_DIRECTORY_PATH = f"{PACKAGE_DIRECTORY}/../{LOG_DIRECTORY}"

file_formatter = logging.Formatter(
    "%(asctime)s %(levelname)s -- %(processName)s -- %(message)s "
)
console_formatter = logging.Formatter("%(levelname)s -- %(processName)s -- %(message)s")

levelmap: Dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARN,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LogListener(Process):
    def __init__(self: "LogListener", queue, levelcode="INFO") -> None:
        super().__init__()
        self.queue = queue
        self.directory = LOG_DIRECTORY_PATH
        level = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARN,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }[levelcode]

        logger = logging.getLogger()
        logger.setLevel(level)

        file_handler = logging.FileHandler(f"{self.directory}/logs.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(level)
        # console_handler.setFormatter(console_formatter)
        # logger.addHandler(console_handler)

    def run(self: "LogListener") -> None:
        while True:
            while not self.queue.empty():
                record = self.queue.get()
                logger = logging.getLogger(record.name)
                logger.handle(record)
            sleep(1)
