import os
import time
import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.log_util import get_logger
from src.utils.config_loader import LOG_DIR


def list_logs():
    pattern = os.path.join(LOG_DIR, "output_*.log")
    return sorted(glob.glob(pattern))


def touch_default_logger():
    logger = get_logger("LogUtilTestDefault")
    logger.debug("default logger debug")
    logger.info("default logger info")
    logger.warning("default logger warning")
    logger.error("default logger error")


def touch_named_logger():
    ts = int(time.time())
    fname = os.path.join(LOG_DIR, f"output_named_{ts}.log")
    logger = get_logger("LogUtilTestNamed", file=fname)
    logger.info("named logger info")
    logger.error("named logger error")
    return fname


def simulate_other_module_usage():
    # 模拟在其他模块中按推荐方式获取并使用 logger
    other_logger = get_logger("SomeOtherModule")
    other_logger.info("other module info")
    other_logger.debug("other module debug")


def main():
    print("LOG_DIR:", LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

    before = set(list_logs())

    # 1) 默认文件名（自动按时间戳生成）
    touch_default_logger()
    time.sleep(0.1)  # 确保文件落盘
    after_default = set(list_logs())
    new_default = sorted(after_default - before)
    assert len(new_default) >= 1, "No new default log file generated"
    print("New default log files:", new_default)

    # 2) 指定文件名
    named_path = touch_named_logger()
    time.sleep(0.1)
    assert os.path.exists(named_path), f"Named log file not found: {named_path}"
    print("Named log file:", named_path)

    # 3) 模拟其他模块使用
    simulate_other_module_usage()
    time.sleep(0.1)

    # 再次列举，确认文件仍然存在且可追加
    final_list = list_logs()
    print("All log files:", final_list)
    print("Test finished OK.")


if __name__ == "__main__":
    main()