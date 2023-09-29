import json
import os
import streamlit as st
from webui import MENU_ITEMS, TTS_MODELS, get_cwd, i18n
from webui.chat import init_assistant_template
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import initial_voice_conversion_params, voice_conversion_form

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
        voice_models=get_voice_list(),
        characters=get_character_list(),
        assistant_template=init_assistant_template(),
        tts_options=initial_voice_conversion_params(),
    )
    return state

def refresh_data(state):
    state.model_list = get_model_list()
    state.voice_models = get_voice_list()
    state.characters = get_character_list()
    return state

def save_character(state):
    character_file = os.path.join(CWD,"models","Characters",f"{state.assistant_template.name}.json")
    with open(character_file,"w") as f:
        loaded_state = {
            "assistant_template": state.assistant_template,
            "tts_options": state.tts_options,
            "voice": state.voice_model,
            "tts_method": state.tts_method
        }
        f.write(json.dumps(loaded_state,indent=2))
    state = refresh_data(state)
    if state.character: state.character.load_character(character_file)
    return state

def load_character(state):
    with open(state.selected_character,"r") as f:
        loaded_state = json.load(f)
        state.assistant_template = ObjectNamespace(**loaded_state["assistant_template"])
        
        state.tts_options = vars(state.tts_options)
        state.tts_options.update(loaded_state["tts_options"])
        state.tts_options = ObjectNamespace(**state.tts_options)
        state.voice_model = loaded_state["voice"]
        state.tts_method = loaded_state["tts_method"]
    state = refresh_data(state)
    return state

def render_model_params_form(state):
    state.model_params.n_ctx = st.slider("Max Context Length", min_value=512, max_value=4096, step=512, value=state.model_params.n_ctx)
    state.model_params.n_gpu_layers = st.slider("GPU Layers", min_value=0, max_value=64, step=4, value=state.model_params.n_gpu_layers)
    state.llm_options.max_tokens = st.slider("New Tokens",min_value=24,max_value=256,step=8,value=state.llm_options.max_tokens)
    return state


def render_tts_options_form(state):

    col1, col2 =st.columns(2)
    state.tts_method = col1.selectbox(
                i18n("tts.model.selectbox"),
                options=TTS_MODELS,
                index=get_index(TTS_MODELS,state.tts_method),
                format_func=lambda option: option.upper()
                )
    state.voice_model = col2.selectbox(
            i18n("inference.voice.selectbox"),
            options=state.voice_models,
            index=get_index(state.voice_models,state.voice_model),
            format_func=lambda option: os.path.basename(option).split(".")[0]
            )
    state.tts_options = voice_conversion_form(state.tts_options)
    return state

def render_assistant_template_form(state):
    state.assistant_template.name = st.text_input("Character Name",value=state.assistant_template.name)
    ROLE_OPTIONS = ["CHARACTER", "USER"]
    state.assistant_template.background = st.text_area("Background", value=state.assistant_template.background, max_chars=1000)
    state.assistant_template.personality = st.text_area("Personality", value=state.assistant_template.personality, max_chars=1000)
    st.write("Example Dialogue")
    state.assistant_template.examples = st.data_editor(state.assistant_template.examples,
                                                        column_order=("role","content"),
                                                        column_config={
                                                            "role": st.column_config.SelectboxColumn("Role",options=ROLE_OPTIONS,required=True),
                                                            "content": st.column_config.TextColumn("Content",required=True)
                                                        },
                                                        use_container_width=True,
                                                        num_rows="dynamic",
                                                        hide_index =True)
    state.assistant_template.greeting = st.text_area("Greeting",value=state.assistant_template.greeting,max_chars=1000)
    return state

def render_character_form(state):
    if not state.selected_character: st.markdown("*Please create a character below if it doesn't exist!*")
    elif st.button("Load Character Info",disabled=not state.selected_character): state=load_character(state)
        
    with st.form("character"):
        template_tab, voice_tab = st.tabs(["Template","Voice"])
        with voice_tab:
            state = render_tts_options_form(state)
        with template_tab:
            state = render_assistant_template_form(state)
        if st.form_submit_button("Save"):
            state = save_character(state)
            st.experimental_rerun()
    
    return state

if __name__=="__main__":
    with SessionStateContext("character_builder",init_state()) as state:
        
        st.title("Character Builder")
        
        if st.button("Refresh Files"):
            state = refresh_data(state)

        # chat settings
        state.selected_character = st.selectbox("Your Character",
                                            options=state.characters,
                                            index=get_index(state.characters,state.selected_character),
                                            format_func=lambda x: os.path.basename(x))
        state = render_character_form(state)