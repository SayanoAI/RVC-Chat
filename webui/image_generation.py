

import hashlib
import io
import json
import os
import random
import subprocess
import time
from typing import Tuple
import numpy as np
import requests
from PIL import Image

from webui import SERVERS, get_cwd
from webui.utils import pid_is_active

CWD = get_cwd()
MAX_INT32 = np.iinfo(np.int32).max
HEADERS = {"Accept": "*", "Content-Type": "application/json"}
DEFAULT_SAMPLER = {
    "steps": 20,
    "cfg": 7.5,
    "name": "dpmpp_2m"
}
SAMPLER_OPTIONS = ["dpmpp_2m","euler_ancestral","dpmpp_sde","dpmpp_2m_sde","dpmpp_3m_sde"]
ORIENTATION_OPTIONS = ["square", "portrait", "landscape"]
STYLE_OPTIONS = ["anime", "realistic"]

def start_server(host="localhost",port=8188):
    if pid_is_active(None if SERVERS["SD"] is None else SERVERS["SD"].get("pid")): return SERVERS["SD"]["pid"]
    root = os.path.join(CWD,".cache","ComfyUI")
    main = os.path.join(root,"main.py")
    cmd = f"python {main} --port={port}"
    process = subprocess.Popen(cmd, cwd=root)
    
    SERVERS["SD"] = {
        "pid": process.pid,
        "url": f"http://{host}:{port}"
    }
    return SERVERS["SD"]["pid"]

def generate_prompt(positive="",subject="",description="",environment="",emotion="",negative="",orientation="square",seed=-1,randomize=False,
                    positive_prefix="masterpiece, best quality",negative_prefix="(worst quality, low quality:1.4)",
                    positive_suffix="",negative_suffix="watermark, (embedding:bad_pictures:1.1)",
                    style="anime",checkpoint=None,scale=1.,steps=20, cfg=7, name="dpmpp_2m",
                    **kwargs):
    # Get a compiler
    from pybars import Compiler
    compiler = Compiler()

    # Compile the template
    workflow = "txt2img-upscale.txt" if scale>1 else "txt2img.txt"
    with open(os.path.join(CWD,"models","SD",".workflows",workflow),"r") as f:
        source = f.read()
    template = compiler.compile(source)

    if orientation.lower()=="portrait":
        width,height=512,768
    elif orientation.lower()=="landscape":
        width,height=768,512
    else:
        width,height=512,512

    if not checkpoint:
        if style.lower()=="realistic": checkpoint="sayano-realistic.safetensors"
        else: checkpoint="sayano-anime.safetensors"

    if scale>2: scale=2.0
    elif scale<1: scale=1.0

    if cfg>12: cfg=12.0
    elif cfg<1: cfg=1.0

    if steps>50: steps=50
    elif steps<20: steps=20

    # Render the template
    sampler = dict(DEFAULT_SAMPLER)
    sampler.update(steps=steps, cfg=cfg, name=name)

    output = template(dict(
        width=width,
        height=height,
        checkpoint=checkpoint,
        scale=scale,
        positive=", ".join(str(i) for i in [
            positive_prefix,positive,subject,description,emotion,environment,style,positive_suffix] if i and len(i)),
        negative=", ".join(str(i) for i in [negative_prefix,negative,negative_suffix] if i and len(i)),
        sampler=dict(
            seed=random.randint(0,MAX_INT32) if seed<0 or randomize else seed,
            **sampler
        )
    ))
    return json.loads(output)

def poll_prompt(prompt: dict, url = None, timeout=60):
    prompt_id = None

    try:
        with requests.post(f"{url}/prompt",json={"prompt": prompt},headers=HEADERS) as req:
            if req.status_code==200:
                result = req.json()
                prompt_id = result["prompt_id"]

        if prompt_id is not None:
            for i in range(timeout):
                time.sleep(1)
                print(f"polling for image data... {i+1}/{timeout}")
                with requests.get(f"{url}/history/{prompt_id}",headers=HEADERS) as req:
                    if req.status_code==200:
                        result = req.json()
                        if prompt_id in result: return result[prompt_id]
    except Exception as e:
        print(e)

    return None

def upload_image(image, url=None):
    
    filename = hashlib.md5(image).hexdigest()
    files = {"image": (filename,image)}

    # check if image exists
    with requests.get(f"{url}/view",stream=True,params=dict(filename=filename,type="input"),headers=HEADERS) as req:
        if req.status_code==200: return filename
        else:
            # upload image
            with requests.post(f"{url}/upload/image",files=files) as req:
                
                if req.status_code==200:
                    result = req.json()
                    img_name = result.get("name")

                    if img_name == filename: return filename
                else: print(req.reason)
    return None

def describe_image(image: bytes, url = None, timeout=60):
    if url is None: url = SERVERS["SD"]["url"]

    # Get a compiler
    from pybars import Compiler
    compiler = Compiler()
    workflow = "img2txt.txt"
    img_name = tags = None

    try:
        img_name = upload_image(image, url)
        
        # Compile the template
        if img_name:
            with open(os.path.join(CWD,"models","SD",".workflows",workflow),"r") as f:
                source = f.read()
            template = compiler.compile(source)
            prompt = template(dict(image=img_name))
            prompt = json.loads(prompt)
            print(prompt)
            # call prompt
            result = poll_prompt(prompt, url=url, timeout=timeout)
            if result: tags = ", ".join(result["outputs"]["2"]["tags"]).replace("_"," ")
    except Exception as e:
        print(e)

    return tags


def generate_images(prompt: dict, url = None, timeout=60):
    if url is None: url = SERVERS["SD"]["url"]
    images = output = []

    try:
        result = poll_prompt(prompt, url=url, timeout=timeout)

        if result:
            images = result["outputs"]["19"]["images"]
            
            for image in images:
                with requests.get(f"{url}/view",stream=True,params=image,headers=HEADERS) as req:
                    if req.status_code==200 and req.content:
                        output.append(Image.open(io.BytesIO(req.content)))

    except Exception as e:
        print(e)
        
    return output