# Import streamlit and requests libraries
import random
from time import sleep
import numpy as np
import streamlit as st

from webui import MENU_ITEMS, SERVERS, ObjectNamespace, get_cwd
from webui.image_generation import generate_images, start_server, generate_prompt
from webui.utils import pid_is_active
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list, image_generation_form, initial_image_generation_state
from webui.contexts import ProgressBarContext, SessionStateContext

CWD = get_cwd()

MAX_INT32 = np.iinfo(np.int32).max

HEADERS = {"Accept": "*", "Content-Type": "application/json"}

if __name__=="__main__":
    with SessionStateContext("comfyui_api",initial_state=initial_image_generation_state()) as state:
        state.remote_bind = st.checkbox("Bind to 0.0.0.0 (Required for docker or remote connections)", value=state.remote_bind)
        state.host = "0.0.0.0" if state.remote_bind else "localhost"
        state.port = st.number_input("Port", value=state.port or 8188)
        state.url = st.text_input("Server URL", value = f"http://{state.host}:{state.port}")
        placeholder = st.container()
        
        is_active = pid_is_active(None if SERVERS["SD"] is None else SERVERS["SD"].get("pid"))
        if st.button("Start Server",disabled=is_active):
            with ProgressBarContext([1]*5,sleep,"Waiting for comfyui to load") as pb:
                start_server(host=state.host,port=state.port)
                pb.run()
                st.experimental_rerun()

        if is_active:
            with st.form("generate"):
                state = image_generation_form(state)

                # Create a button to submit the prompt and generate the image
                if st.form_submit_button("Generate"):
                    if state.randomize or state.seed<0:
                        state.seed=random.randint(0,MAX_INT32)
                    prompt = generate_prompt(**state)
                    state.images = generate_images(prompt=prompt)
                    st.experimental_rerun()
            
            if state.images:
                st.image(state.images)

        active_subprocess_list()