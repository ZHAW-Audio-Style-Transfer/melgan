"""
Logs prettified information to console and output file.
"""

from helpers.generic import ensureFolderExists, get_timestamp
from config.config import Config
import colorama
import time as sys_time
from .colors import Colors
from pathlib import Path


class Logger(object):

    timers = {}
    log_file = ""
    write_file = False

    @staticmethod
    def initialize(write_file=True):
        Logger.write_file = write_file
        Logger.timers = {}
        if write_file:
            Logger.log_file = Path(
                Config().environment.output_path,
                Config().environment.log_file,
            )
        colorama.init()

    @staticmethod
    def format(iterable):
        return "".join(str(i) for i in iterable)

    @staticmethod
    def log(*args):
        print(Colors.GREEN + Logger.format(args) + Colors.END)
        Logger._write_log("Log", Logger.format(args))

    @staticmethod
    def warn(*args):
        print(
            Colors.YELLOW + "WARN:\t" + Colors.END + Colors.MAGENTA,
            Logger.format(args),
            Colors.END,
        )
        Logger._write_log("Warning", Logger.format(args))

    @staticmethod
    def error(*args):
        print(
            Colors.RED + Colors.BLINK + "ERROR:\t" + Colors.END + Colors.RED,
            Logger.format(args),
            Colors.END,
        )
        Logger._write_log("Error", Logger.format(args))

    @staticmethod
    def time(key):
        Logger.timers[key] = sys_time.time()

    @staticmethod
    def time_end(key):
        if key in Logger.timers:
            t = sys_time.time() - Logger.timers[key]
            print("\t" + str(t) + Colors.DIM + " s \t" + key + Colors.END)
            del Logger.timers[key]

    @staticmethod
    def notify(*args):
        # Play bell
        print("\a")

    @staticmethod
    def _write_log(title, message):
        if Logger.write_file:
            try:
                if not Logger.log_file:
                    return
                if title:
                    content = "%s\t%s: %s" % (get_timestamp(), title, message)
                else:
                    content = "%s\t%s" % (get_timestamp(), message)

                ensureFolderExists(Logger.log_file.parent)
                with open(Logger.log_file, "a", newline="\n", encoding="utf8") as file:
                    file.write(content + "\n")
            except:
                pass
