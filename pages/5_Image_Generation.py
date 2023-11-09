# Import streamlit and requests libraries
import html
import json
import os
import random
import subprocess
from time import sleep
import time
import numpy as np
import streamlit as st
import requests

from webui import MENU_ITEMS, SERVERS, ObjectNamespace, get_cwd
from webui.image_generation import start_server
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list, st_iframe
from webui.contexts import ProgressBarContext, SessionStateContext

CWD = get_cwd()

MAX_INT32 = np.iinfo(np.int32).max

DEFAULT_SAMPLER = {
    "steps": 20,
    "cfg": 7.5,
    "name": "dpmpp_2m"
}

HEADERS = {"Accept": "*", "Content-Type": "application/json"}

@st.cache_data
def generate_prompt(positive,negative="",width=512,height=512,seed=-1,**sampler):
    # Get a compiler
    from pybars import Compiler
    compiler = Compiler()

    # Compile the template
    with open(os.path.join(CWD,"models","SD",".workflows","txt2img.txt"),"r") as f:
        source = f.read()
    template = compiler.compile(source)

    # Render the template
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

@st.cache_data
def generate_images(url: str, prompt: dict, timeout=60):
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
        st.toast(f"Failed to generate: {e}")
        
    return output

def initial_state():
    return ObjectNamespace(
        positive="",
        negative="(worst quality, low quality:1.4)",
        seed=-1,
        width=512,
        height=512,
        prompt="",
        steps=20,
        cfg=7.5,
        name="dpmpp_2m"
    )
if __name__=="__main__":
    with SessionStateContext("comfyui_api",initial_state=initial_state()) as state:
        state.remote_bind = st.checkbox("Bind to 0.0.0.0 (Required for docker or remote connections)", value=state.remote_bind)
        state.host = "0.0.0.0" if state.remote_bind else "localhost"
        state.port = st.number_input("Port", value=state.port or 8188)
        state.url = st.text_input("Server URL", value = f"http://{state.host}:{state.port}")
        placeholder = st.container()
        
        if st.button("Start Server",disabled=SERVERS["SD"] is not None):
            with ProgressBarContext([1]*5,sleep,"Waiting for comfyui to load") as pb:
                start_server(host=state.host,port=state.port)
                pb.run()
                st.experimental_rerun()

        active_subprocess_list()

        with st.form("generate"):
            # Create a text input for the user to enter a prompt
            state.positive = st.text_area("Enter your positive prompt for image generation",value=state.positive)
            state.negative = st.text_area("Enter your negative prompt for image generation",value=state.negative)
            c1, c2, c3 = st.columns(3)
            state.width = c1.number_input("Width",min_value=512,max_value=1024,step=4,value=state.width)
            state.height = c2.number_input("Height",min_value=512,max_value=1024,step=4,value=state.height)
            state.seed = c3.number_input("Seed",min_value=-1,max_value=MAX_INT32,step=1,value=state.seed)
            state.steps = c1.number_input("Steps",min_value=1,max_value=100,step=1,value=state.steps)
            state.cfg = c2.number_input("CFG",min_value=0.,max_value=15.,step=.1,value=state.cfg)
            state.name = c3.selectbox("Sampler Name",options=["dpmpp_2m"],index=0)

            # Create a button to submit the prompt and generate the image
            if st.form_submit_button("Generate"):
                state.prompt = generate_prompt(state.positive,negative=state.negative,seed=state.seed,**DEFAULT_SAMPLER)
                state.images = generate_images(state.url,prompt=state.prompt)
        
        if state.images:
            st.image(state.images)

        if SERVERS["SD"] is not None: st_iframe(state.url,height=1024)

