"""
loads a configuration from 3 files, high priority overwrites low priority:
1. A config.yml found in the current working directory (high priority)
2. ~/.dimmilitho.yml specific for the machine
3. the default config is in this file
"""

__all__ = ["CONFIG"]

import logging
import pathlib

import hiyapyco

default_config = """
keySample: valueSample
"""


cwd = pathlib.Path.cwd()
# cwd_config = cwd / "config.yml"
# home = pathlib.Path.home()
# home_config = home / "config.yml"
module_path = pathlib.Path(__file__).parent.absolute()
repo_path = module_path.parent
module_config = module_path / "config.yml"

CONFIG = hiyapyco.load(
    str(default_config),
    str(module_config),
    failonmissingfiles=True,
    loglevelmissingfiles=logging.DEBUG,
)

CONFIG["module_path"] = str(module_path)
CONFIG["repo_path"] = str(repo_path)




# if __name__ == "__main__":
#     print(CONFIG["repo"])