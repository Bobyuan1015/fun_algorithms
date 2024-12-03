import logging
import os
from datetime import datetime

from confs.path_conf import system_log_dir


class Logger:
    def __init__(self, pre_dir,pre_file):
        # 设置 logging 配置
        self.logger = logging.getLogger(pre_dir)
        self.logger.setLevel(logging.INFO)
        # 获取当前日期和时间
        now = datetime.now()

        # 将当前日期格式化为字符串
        date_string = now.strftime("%Y-%m-%d")
        base_dir = f'{system_log_dir}/{pre_dir}'
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
            print(f"Directory '{base_dir}' created.")
        else:
            print(f"Directory '{base_dir}' already exists.")
        # 创建文件处理器
        file_handler = logging.FileHandler(f"{base_dir}/{pre_file}_{date_string}.log")
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)

        # 创建格式器并将其添加到处理器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        # 将处理器添加到 logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # 示例输出
        self.logger.info("开始启动。")

    def get_logger(self):
        return self.logger

# 使用示例
if __name__ == "__main__":
    my_logger = Logger("my_log").get_logger()
