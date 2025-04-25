from random import choice
import random
import logging
import os
import sys
from datetime import datetime

class AlertSystem:
    def __init__(self, module_name="unknown", real_mode=False, log_to_file=True):
        self.module_name = module_name
        self.real_mode = real_mode
        self.log_to_file = log_to_file
        self.log_path = os.path.join("logs", f"{module_name}.log")
        os.makedirs("logs", exist_ok=True)

        self.logger = logging.getLogger(f"AlertSystem.{module_name}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if log_to_file:
            file_handler = logging.FileHandler(self.log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.info("Alert System initialized")

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def success(self, message):
        self.logger.info(f"✅ {message}")

    def debug(self, message):
        self.logger.debug(message)

    def log(self, level, message):
        self.logger.log(level, message)
