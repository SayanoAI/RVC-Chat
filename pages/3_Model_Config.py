import json
import os
import streamlit as st
from webui import MENU_ITEMS, get_cwd
from webui.chat import init_llm_options, init_model_config, init_model_data, init_model_params
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from lib.model_utils import get_hash

from webui.contexts import SessionStateContext

from webui.utils import ObjectNamespace

from webui.utils import get_filenames, get_index

CWD = get_cwd()

def get_model_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="LLM",exts=["bin","gguf"])]
    return models_list

def get_voice_list():
    models_list = [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="RVC",exts=["pth"])]
    return models_list

def get_character_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="Characters",exts=["json"])]
    return models_list

def init_state():
    state = ObjectNamespace(
        model_config=init_model_config(),
        model_params=init_model_params(),
        model_list=get_model_list(),
        llm_options=init_llm_options(),
    )
    return state

def refresh_data(state):
    state.model_list = get_model_list()
    return state

def save_model_config(state):
    try:
        fname = os.path.join(CWD,"models","LLM","config.json")
        key = get_hash(os.path.join(CWD,"models","LLM",os.path.basename(state.model_params.fname)))

        if os.path.isfile(fname):
            with open(fname,"r") as f:
                data = json.load(f)
        else:
            data = {}

        with open(fname,"w") as f:
            data[key] = {
                "version": 2,
                "params": state.model_params,
                "config": state.model_config,
                "options": state.llm_options,
            }
            f.write(json.dumps(data,indent=2))
        state = refresh_data(state)
        st.toast(f"Successfully saved config: {key}")
    except Exception as e:
        st.toast(f"Failed to save config: {e}")
    return state

def load_model_config(state):
    fname = os.path.join(CWD,"models","LLM","config.json")
    key = get_hash(state.selected_llm)

    with open(fname,"r") as f:
        data = json.load(f) if os.path.isfile(fname) else {}    
        model_data = data[key] if key in data else init_model_data()

        if "version" in model_data and model_data["version"]==2:
            state.model_params = ObjectNamespace(**model_data["params"])
            state.model_config = ObjectNamespace(**model_data["config"])
            state.llm_options = ObjectNamespace(**model_data["options"])
        else: # old version
            state.model_config.prompt_template = model_data["prompt_template"]
            state.model_config.chat_template = model_data["chat_template"]
            state.model_config.instruction = model_data["instruction"]
            state.model_config.mapper = model_data["mapper"]
            state.model_params.n_ctx = model_data["n_ctx"]
            state.model_params.n_gpu_layers = model_data["n_gpu_layers"]
            state.llm_options.max_tokens = model_data["max_tokens"]
    state = refresh_data(state)
    return state

def render_model_config_form(state):
    state.model_config.instruction = st.text_area("Instruction",value=state.model_config.instruction)
    state.model_config.chat_template = st.text_area("Dialogue Format",value=state.model_config.chat_template)
    state.model_config.prompt_template = st.text_area("Prompt Template",value=state.model_config.prompt_template,height=400)
    state.model_config.mapper = st.data_editor(state.model_config.mapper,
                                                        column_order=("_index","value"),
                                                        use_container_width=False,
                                                        num_rows="fixed",
                                                        disabled=["_index"],
                                                        hide_index=False)
    return state

def render_model_params_form(state):
    state.model_params.n_ctx = st.slider("Max Context Length", min_value=512, max_value=4096, step=512, value=state.model_params.n_ctx)
    state.model_params.n_gpu_layers = st.slider("GPU Layers", min_value=0, max_value=64, step=4, value=state.model_params.n_gpu_layers)
    state.llm_options.max_tokens = st.slider("New Tokens",min_value=24,max_value=256,step=8,value=state.llm_options.max_tokens)
    return state

def render_llm_form(state):
    if not state.selected_llm: st.markdown("*Please save your model config below if it doesn't exist!*")
    elif st.button("Load LLM Config",disabled=not state.selected_llm): state=load_model_config(state)
    
    with st.form("model.loader"):
        state = render_model_params_form(state)
        state = render_model_config_form(state)

        if st.form_submit_button("Save Configs",disabled=not state.selected_llm):
            state = save_model_config(state)
    return state

if __name__=="__main__":
    with SessionStateContext("llm_config",init_state()) as state:
        
        st.title("Model Configuration")

        if st.button("Refresh Files"):
            state = refresh_data(state)

        # chat settings
        state.selected_llm = st.selectbox("Choose a language model",
                            options=state.model_list,
                            index=get_index(state.model_list,state.selected_llm),
                            format_func=lambda x: os.path.basename(x))
        state = render_llm_form(state)