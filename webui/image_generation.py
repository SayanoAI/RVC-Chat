

import hashlib
import io
import json
import os
import random
import shutil
import subprocess
import time
import numpy as np
import requests
from PIL import Image

from webui import SERVERS, get_cwd, config
from webui.utils import  pid_is_active
from pybars import Compiler

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
NSFW_KEYWORDS = ["nsfw", "nude", "loli", "naked", "sex", "pussy", "penis"]

def start_server(host="localhost",port=8188):
    pid = SERVERS.SD_PID
    if pid_is_active(pid): return SERVERS.SD_URL
    root = os.path.join(CWD,".cache","ComfyUI")
    shutil.copy(os.path.join(CWD,"extra_model_paths.yaml"),root) # copy configs
    main = os.path.join(root,"main.py")
    cmd = [config.python_cmd, main, f"--port={port}"]
    process = subprocess.Popen(cmd, cwd=root)
    
    SERVERS.SD_PID = process.pid
    SERVERS.SD_URL = f"http://{host}:{port}"

    return SERVERS.SD_URL

def generate_prompt(checkpoint: str=None, lora: str=None,width=512,height=512,
                    positive="",subject="",description="",environment="",emotion="",negative="",
                    positive_prefix="masterpiece, best quality",negative_prefix="(worst quality, low quality:1.4)",
                    positive_suffix="",negative_suffix="watermark, (embedding:bad_pictures:1.1)",censor=False,
                    style="",scale=1.,steps=20, cfg=7, name="dpmpp_2m",orientation="square",seed=-1,randomize=False,
                    image: bytes=None, change_ratio="small", *args,**kwargs):
    
    # error check numeric inputs
    scale = float(scale)
    cfg = float(cfg)
    steps = int(steps)
    seed = int(seed)
    if scale>2: scale=2.0
    elif scale<1: scale=1.0
    change_ratio = change_ratio.lower()

    if change_ratio=="large": denoise=.88
    elif change_ratio=="medium": denoise=.69
    elif change_ratio=="small": denoise=.5
    else: denoise=.95

    if cfg>12: cfg=12.0
    elif cfg<1: cfg=1.0

    if steps>50: steps=50
    elif steps<20: steps=20

    # Compile the template
    workflow = "txt2img-upscale.txt" if scale>1 else "txt2img.txt"

    # upload image
    if type(image)==bytes: image = upload_image(image)
    if image: workflow = "img-upscale.txt" if scale>1 else "img2img.txt"
    else: workflow = "txt2img-upscale.txt" if scale>1 else "txt2img.txt"

    # Get a compiler
    compiler = Compiler()
    with open(os.path.join(CWD,"models","SD",".workflows",workflow),"r") as f:
        source = f.read()
    template = compiler.compile(source)

    if workflow == "img-upscale.txt":
        width, height = int(width * scale), int(height * scale)
    else:
        if orientation.lower()=="portrait":
            width,height=512,768
        elif orientation.lower()=="landscape":
            width,height=768,512
        else:
            width,height=512,512
    
    # Render the template
    sampler = dict(DEFAULT_SAMPLER)
    sampler.update(steps=steps, cfg=cfg, name=name, denoise=denoise)
    nsfw_negative = [neg for neg in NSFW_KEYWORDS if censor]

    output = template(dict(
        image=image,
        width=width,
        height=height,
        checkpoint=checkpoint,
        lora=lora,
        scale=scale,
        positive=", ".join(str(i) for i in [
            positive_prefix,positive,subject,description,emotion,environment,style,positive_suffix] if i and len(i)),
        negative=", ".join(str(i) for i in [*nsfw_negative,negative_prefix,negative,negative_suffix] if i and len(i)),
        sampler=dict(
            seed=random.randint(0,MAX_INT32) if seed<0 or randomize else seed,
            **sampler
        )
    ))
    print(f"{output}")
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

def upload_image(image: bytes, url=None):
    if url is None: url = SERVERS.SD_URL

    filename = hashlib.md5(image).hexdigest() + ".png"
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
                else: print(f"upload status {req.status_code}: {req.reason}")
    return None

def describe_image(image: bytes, url = None, timeout=60):
    if url is None: url = SERVERS.SD_URL

    # Get a compiler
    from pybars import Compiler
    compiler = Compiler()
    workflow = "img2txt.txt"
    img_name = tags = None

    try:
        img_name = upload_image(image, url)
        
        if img_name:
            with open(os.path.join(CWD,"models","SD",".workflows",workflow),"r") as f:
                source = f.read()
            template = compiler.compile(source)
            prompt = template(dict(image=img_name))
            prompt = json.loads(prompt)

            # call prompt
            result = poll_prompt(prompt, url=url, timeout=timeout)
            if result: tags = ", ".join(result["outputs"]["2"]["tags"]).replace("_"," ")
    except Exception as e:
        print(e)

    return tags

def generate_images(prompt: dict=None, url = None, timeout=60, *args, **kwargs):
    if url is None: url = SERVERS.SD_URL
    if prompt is None: prompt = generate_prompt(*args,**kwargs)

    images = output = []

    try:
        result = poll_prompt(prompt, url=url, timeout=timeout)

        if result:
            output_keys = [k for k,v in prompt.items() if v.get("class_type")=="PreviewImage"]

            for key in output_keys:
                images = result["outputs"][key]["images"]
                
                for image in images:
                    with requests.get(f"{url}/view",stream=True,params=image,headers=HEADERS) as req:
                        if req.status_code==200 and req.content:
                            output.append(Image.open(io.BytesIO(req.content)))

    except Exception as e:
        print(e)
        
    return output