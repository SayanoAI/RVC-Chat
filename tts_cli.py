import hashlib
import numpy as np
import os
from lib.infer_pack.text.cleaners import english_cleaners
from webui import get_cwd

from webui.audio import load_input_audio
from webui.downloader import BASE_CACHE_DIR

CWD = get_cwd()
    
os.makedirs(os.path.join(CWD,"models","TTS","embeddings"),exist_ok=True)
TTS_MODELS_DIR = os.path.join(CWD,"models","TTS")
DEFAULT_SPEAKER = os.path.join(TTS_MODELS_DIR,"embeddings","Sayano.npy")

def cast_to_device(tensor, device):
    try:
        return tensor.to(device)
    except Exception as e:
        print(e)
        return tensor


def __edge__(text, speaker="en-US-JennyNeural"):
    import edge_tts
    import asyncio
    from threading import Thread
    temp_dir = os.path.join(BASE_CACHE_DIR,"tts","edge",speaker)
    os.makedirs(temp_dir,exist_ok=True)
    tempfile = os.path.join(temp_dir,f"{hashlib.md5(text.encode('utf-8')).hexdigest()}.wav")

    async def fetch_audio():
        communicate = edge_tts.Communicate(text, speaker)

        try:
            with open(tempfile, "wb") as data:
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        data.write(chunk["data"])
        except Exception as e:
            print(e)
    
    thread = Thread(target=asyncio.run, args=(fetch_audio(),),name="edge-tts",daemon=True)
    thread.start()
    thread.join()
    
    try:
        audio, sr = load_input_audio(tempfile,sr=16000)
        return audio, sr
    except Exception as e:
        print(e)
        return None


def generate_speech(text, method="edge",speaker="en-US-JennyNeural",dialog_only=False):
    
    text = english_cleaners(text.strip(),dialog_only=dialog_only) #clean text
    if text and len(text) == 0:
        return (np.zeros(0).astype(np.int16),16000)
    
    if method=="edge":
        return __edge__(text,speaker)
    else: return None