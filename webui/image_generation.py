

from functools import lru_cache
import json
import os
import random
import subprocess
import time
import numpy as np
import requests

from webui import SERVERS, get_cwd

CWD = get_cwd()
MAX_INT32 = np.iinfo(np.int32).max
HEADERS = {"Accept": "*", "Content-Type": "application/json"}
DEFAULT_SAMPLER = {
    "steps": 20,
    "cfg": 7.5,
    "name": "dpmpp_2m"
}

def start_server(host="localhost",port=8188):
    if SERVERS["SD"] and "pid" in SERVERS["SD"]: return SERVERS["SD"]["pid"]
    main = os.path.join(CWD,".cache","ComfyUI","main.py")
    cmd = f"python {main} --port={port} --listen"
    
    SERVERS["SD"] = {
        "pid": subprocess.Popen(cmd, shell=True, cwd=CWD).pid,
        "url": f"http://{host}:{port}"
    }
    return SERVERS["SD"]["pid"]

@lru_cache
def generate_prompt(positive,negative="",width=512,height=512,seed=-1,**kwargs):
    # Get a compiler
    from pybars import Compiler
    compiler = Compiler()

    # Compile the template
    with open(os.path.join(CWD,"models","SD",".workflows","txt2img.txt"),"r") as f:
        source = f.read()
    template = compiler.compile(source)

    # Render the template
    sampler = dict(DEFAULT_SAMPLER)
    sampler.update(kwargs)

    output = template(dict(
        width=width,
        height=height,
        positive=positive,
        negative=negative,
        sampler=dict(
            seed=random.randint(0,MAX_INT32) if seed<0 else seed,
            **sampler
        )
    ))

    return json.loads(output)

def generate_images(prompt: dict, url = None, timeout=60):
    if url is None: url = SERVERS["SD"]["url"]
    prompt_id = None
    images = output = []

    try:
        with requests.post(f"{url}/prompt",json={"prompt": prompt},headers=HEADERS) as req:
            if req.status_code==200:
                result = req.json()
                prompt_id = result["prompt_id"]

        if prompt_id is not None:
            for i in range(timeout):
                print(f"polling for image data... {i+1}")
                time.sleep(1)
                with requests.get(f"{url}/history/{prompt_id}",headers=HEADERS) as req:
                    if req.status_code==200:
                        result = req.json()
                        if prompt_id in result:
                            images = result[prompt_id]["outputs"]["19"]["images"]
                            break
            
            for image in images:
                with requests.get(f"{url}/view",stream=True,params=image,headers=HEADERS) as req:
                    if req.status_code==200:
                        output.append(req.content)

    except Exception as e:
        print(e)
        
    return output
