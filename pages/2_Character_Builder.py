import json
import os
import random
import streamlit as st
from webui import MENU_ITEMS, TTS_MODELS, ObjectNamespace, get_cwd, i18n
from webui.chat import init_assistant_template, init_tts_options, load_character_data
from webui.image_generation import MAX_INT32, generate_images, generate_prompt
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import image_generation_form, initial_image_generation_state, voice_conversion_form

from webui.contexts import SessionStateContext

from webui.utils import get_filenames, get_index

CWD = get_cwd()

def get_voice_list():
    models_list = [""] + [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="RVC",exts=["pth"])]
    return models_list

def get_character_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models"),folder="Characters",exts=["json"])]
    return models_list

def init_state():
    state = ObjectNamespace(
        voice_models=get_voice_list(),
        characters=get_character_list(),
        assistant_template=init_assistant_template(),
        tts_options=init_tts_options(),
        preview=None
    )
    return state

def refresh_data(state):
    state.voice_models = get_voice_list()
    state.characters = get_character_list()
    return state

def save_character(state):
    try:
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
        st.toast(f"Successfully saved character: {character_file}")
    except Exception as e:
        st.toast(f"Failed to save character: {e}")
    return state

def load_character(state):
    loaded_state = load_character_data(state.selected_character)
    state.assistant_template = ObjectNamespace(**loaded_state["assistant_template"])
    state.tts_options = ObjectNamespace(**loaded_state["tts_options"])
    state.voice_model = loaded_state["voice"]
    state.tts_method = loaded_state["tts_method"]
    state = refresh_data(state)
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
    state.assistant_template.background = st.text_area("Background", value=state.assistant_template.background)
    state.assistant_template.personality = st.text_area("Personality", value=state.assistant_template.personality)
    # state.assistant_template.appearance = st.text_area("Appearance", value=state.assistant_template.appearance)
    st.write(f"Appearance: {json.dumps(state.assistant_template.appearance)}")
    state.assistant_template.scenario = st.text_area("Scenario", value=state.assistant_template.scenario)
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
    state.assistant_template.greeting = st.text_area("Greeting",value=state.assistant_template.greeting)
    return state

def render_character_form(state):
    if not state.selected_character: st.markdown("*Please create a character below if it doesn't exist!*")
    elif st.button("Load Character Info",disabled=not state.selected_character): state=load_character(state)
        
    with st.expander("Customize your appearance",expanded=state.preview is None):
        if type(state.assistant_template.appearance)==str:
            default_state = initial_image_generation_state()
            default_state["description"] = state.assistant_template.appearance
            state.assistant_template.appearance = default_state
        
        state.assistant_template.appearance = image_generation_form(ObjectNamespace(**state.assistant_template.appearance))
        if st.button("Generate"):
            if state.assistant_template.appearance.randomize or state.assistant_template.appearance.seed<0:
                state.assistant_template.appearance.seed=random.randint(0,MAX_INT32)
            prompt = generate_prompt(**state.assistant_template.appearance)
            state.preview = generate_images(prompt=prompt)
            st.experimental_rerun()

    if state.preview: st.image(state.preview)

    with st.form("character"):
        template_tab, voice_tab = st.tabs(["Template","Voice"])
        with voice_tab:
            state = render_tts_options_form(state)
        with template_tab:
            state = render_assistant_template_form(state)
        if st.form_submit_button("Save",type="primary"):
            state = save_character(state)
    
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
