import os
from time import sleep
from webui import MENU_ITEMS, SERVERS, ObjectNamespace, get_cwd
import streamlit as st
from webui.chat import load_model_data
from webui.kobold_cpp import start_server

from webui.utils import get_filenames, get_index, pid_is_active

st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list, st_iframe
from webui.contexts import ProgressBarContext, SessionStateContext

CWD = get_cwd()

def get_model_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="LLM",exts=["bin","gguf"])]
    return models_list

def render_model_params_form(state):
    CONTEXTSIZE_OPTIONS = [256,512,1024,2048,3072,4096,6144,8192,12288,16384,24576,32768,65536]
    state.n_ctx = st.select_slider("Max Context Length", options=CONTEXTSIZE_OPTIONS, value=state.n_ctx)
    state.n_gpu_layers = st.slider("GPU Layers", min_value=0, max_value=64, step=4, value=state.n_gpu_layers)
    
    return state

def initial_state():
    return ObjectNamespace(
        model=None,
        n_ctx=2048,
        n_gpu_layers=0
    )

@st.cache_data
def get_params(model):
    data = load_model_data(model)
    return data["params"]

if __name__=="__main__":
    with SessionStateContext("llm_api",initial_state()) as state:
        is_active = pid_is_active(None if SERVERS["LLM"] is None else SERVERS["LLM"].get("pid"))

        state.model = st.selectbox("Choose a language model",
            options=get_model_list(),
            index=get_index(get_model_list(),state.model),
            format_func=lambda x: os.path.basename(x))
        state.params = get_params(state.model)
        st.write(state.params)

        with st.form("LLM form"):
            state.remote_bind = st.checkbox("Bind to 0.0.0.0 (Required for docker or remote connections)", value=state.remote_bind)
            state.host = "0.0.0.0" if state.remote_bind else "localhost"
            state.port = st.number_input("Port", value=state.port or 8000)
            state.url = st.text_input("Server URL", value = f"http://{state.host}:{state.port}/api")
            state.params = render_model_params_form(state.params)

            if st.form_submit_button("Start Server",disabled=is_active):
                with ProgressBarContext([1]*5,sleep,"Waiting for koboldcpp to load") as pb:
                    start_server(state.model,
                                 host=state.host,
                                 port=state.port,
                                 n_gpu_layers=state.params.n_gpu_layers,
                                 n_ctx=state.params.n_ctx)
                    pb.run()
                    st.experimental_rerun()
                
        active_subprocess_list()
        
        if is_active: st_iframe(url=SERVERS["LLM"]["url"],height=800)