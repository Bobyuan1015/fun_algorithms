import logging
import os
from datetime import datetime

from confs.path_conf import system_log_dir


class Logger:
    def __init__(self, pre_dir,pre_file):
        self.logger = logging.getLogger(pre_dir)
        self.logger.setLevel(logging.INFO)
        now = datetime.now()

        date_string = now.strftime("%Y-%m-%d")
        base_dir = f'{system_log_dir}/{pre_dir}'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"Directory '{base_dir}' created.")
        else:
            print(f"Directory '{base_dir}' already exists.")
        file_handler = logging.FileHandler(f"{base_dir}/{pre_file}_{date_string}.log")
        file_handler.setLevel(logging.INFO)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        self.logger.info("开始启动。")

    def get_logger(self):
        return self.logger

# 使用示例
if __name__ == "__main__":
    my_logger = Logger("my_log").get_logger()
