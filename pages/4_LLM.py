import subprocess
from webui import MENU_ITEMS, get_cwd
import streamlit as st

from webui.utils import get_cache
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list, st_iframe
from webui.contexts import SessionStateContext

CWD = get_cwd()
SERVER = get_cache("LLM")

if __name__=="__main__":
    with SessionStateContext("llm_api") as state:
        state.remote_bind = st.checkbox("Bind to 0.0.0.0 (Required for docker or remote connections)", value=state.remote_bind)
        state.host = "0.0.0.0" if state.remote_bind else "localhost"
        state.port = st.number_input("Port", value=state.port or 8000)
        state.url = st.text_input("Server URL", value = f"http://{state.host}:{state.port}/api")
        active_subprocess_list()
        if state.url:
            st_iframe(url=state.url,height=800)
        
