from typing import TYPE_CHECKING

import json

import logging

if TYPE_CHECKING:
    from tellodrone.core import TelloDrone

class ColourFormatter(logging.Formatter):
    magenta = "\x1b[95;20m"
    blue = "\x1b[96;20m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: magenta + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,   
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }


    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(self: "TelloDrone") -> None:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.NOTSET)
    stream_handler.setFormatter(ColourFormatter('%(asctime)s - %(levelname)s - %(message)s'))

    file_handler = logging.FileHandler(self.log_info_file)
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    self.logger = logging.getLogger()
    self.logger.addHandler(stream_handler)
    self.logger.addHandler(file_handler)
    self.logger.setLevel(logging.NOTSET)
    
    open(self.log_pos_file, "x").close()
    open(self.log_config_file, "x").close()


def save_log_config(self: "TelloDrone") -> None:
    config = {
        "takeoff_pos": self.takeoff_pos.to_arr(),
        "start_pos": self.start_pos.to_arr(),
        "end_pos": self.cur_pos.to_arr(),
        "target_pos": self.target_pos.to_arr(),
        "obstacles": [(obp.to_arr(), obr) for obp, obr in self.obstacles]
    }
    
    with open(self.log_config_file, "w+") as f:
        f.write(json.dumps(config, indent=4))