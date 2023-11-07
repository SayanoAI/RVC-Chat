import subprocess
import random
import html
import json
from webui import MENU_ITEMS, get_cwd
import streamlit as st
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.components import active_subprocess_list
from webui.contexts import SessionStateContext

CWD = get_cwd()

## Ported from streamlit_tensorboard with modifications
def st_swagger(url="http://localhost:8000", width=None, height=None, scrolling=False):
    """Embed kobold server within a Streamlit app
    Parameters
    ----------
    url: string
        URL of the server. Defaults to `http://localhost:8000`
    width: int
        The width of the frame in CSS pixels. Defaults to container width.
    height: int
        The height of the frame in CSS pixels. Defaults to container height.
    scrolling: bool
        If True, show a scrollbar when the content is larger than the iframe.
        Otherwise, do not show a scrollbar. Defaults to False.

    Example
    -------
    >>> st_swagger(url="http://localhost:8000", width=1080)
    """

    frame_id = "swagger-frame-{:08x}".format(random.getrandbits(64))
    shell = """
        <iframe id="%HTML_ID%" width="%WIDTH%" height="%HEIGHT%" frameborder="0">
        </iframe>
        <script>
        (function() {
            const frame = document.getElementById(%JSON_ID%);
            frame.src = new URL(%URL%, window.location);
        })();
        </script>
    """

    replacements = [
        ("%HTML_ID%", html.escape(frame_id, quote=True)),
        ("%JSON_ID%", json.dumps(frame_id)),
        ("%HEIGHT%", str(height) if height else "100%"),
        ("%WIDTH%", str(width) if width else "100%"),
        ("%URL%", json.dumps(url)),
    ]

    for (k, v) in replacements:
        shell = shell.replace(k, v)

    return st.components.v1.html(shell, width=width, height=height, scrolling=scrolling)
    
def start_tensorboard(logdir, host="localhost"):
    cmd = f"tensorboard --logdir={logdir} --host={host}"
    p = subprocess.Popen(cmd, shell=True, cwd=CWD)
    return p

if __name__=="__main__":
    with SessionStateContext("llm_api") as state:
        state.remote_bind = st.checkbox("Bind to 0.0.0.0 (Required for docker or remote connections)", value=state.remote_bind)
        state.host = "0.0.0.0" if state.remote_bind else "localhost"
        state.port = st.number_input("Port", value=state.port or 8000)
        state.url = st.text_input("Server URL", value = f"http://{state.host}:{state.port}/api")
        if state.url:
            st_swagger(url=state.url,height=800)
        placeholder = st.container()
        
        # if st.button("Start Tensorboard", disabled=tensorboard_is_active):
        #     with ProgressBarContext([1]*5,sleep,"Waiting for tensorboard to load") as pb:
        #         start_tensorboard(state.logdir, "localhost" if not state.remote_bind else "0.0.0.0")
        #         pb.run()
        #         st.experimental_rerun()

        active_subprocess_list()
