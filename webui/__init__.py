from functools import lru_cache
import os
import sys
from config import Config
from i18n import I18nAuto


MENU_ITEMS = {
    "Get help": "https://github.com/SayanoAI/RVC-Chat/discussions",
    "Report a Bug": "https://github.com/SayanoAI/RVC-Chat/issues",
    "About": """This project provides a comprehensive platform for training RVC models and generating AI voice covers.
    Check out this github for more info: https://github.com/SayanoAI/RVC-Chat
    """
}

DEVICE_OPTIONS = ["cpu","cuda"]
PITCH_EXTRACTION_OPTIONS = ["crepe","rmvpe","mangio-crepe","rmvpe+"]
TTS_MODELS = ["edge","speecht5","tacotron2"]
N_THREADS_OPTIONS=[1,2,4,8,12,16]
SR_MAP = {"40k": 40000, "48k": 48000}

@lru_cache
def load_config():
    return Config(), I18nAuto()

@lru_cache
def get_cwd():
    CWD = os.getcwd()
    if CWD not in sys.path:
        sys.path.append(CWD)
    return CWD

config, i18n = load_config()