# First
from io import BytesIO
import os
import platform
from pytube import YouTube
import streamlit as st
from webui import MENU_ITEMS, get_cwd
st.set_page_config("RVC Chat",layout="centered",menu_items=MENU_ITEMS)


from tts_cli import STT_MODELS_DIR, stt_checkpoint, load_stt_models

from webui.components import file_downloader, file_uploader_form

from webui.downloader import BASE_MODELS, BASE_MODELS_DIR, LLM_MODELS, RVC_DOWNLOAD_LINK, RVC_MODELS, VITS_MODELS, download_link_generator

CWD = get_cwd()

from webui.contexts import ProgressBarContext

def download_audio_to_buffer(url):
    buffer = BytesIO()
    youtube_video = YouTube(url)
    audio = youtube_video.streams.get_audio_only()
    default_filename = audio.default_filename
    audio.stream_to_buffer(buffer)
    return default_filename, buffer

def render_download_ffmpeg(lib_name="ffmpeg.exe"):
    col1, col2 = st.columns(2)
    is_downloaded = os.path.exists(lib_name)
    col1.checkbox(os.path.basename(lib_name),value=is_downloaded,disabled=True)
    if col2.button("Download",disabled=is_downloaded,key=lib_name):
        link = f"{RVC_DOWNLOAD_LINK}ffmpeg.exe"
        file_downloader((lib_name,link))
        st.experimental_rerun()

def render_download_koboldcpp(lib_name="koboldcpp.exe"):
    col1, col2 = st.columns(2)
    is_downloaded = os.path.exists(lib_name)
    col1.checkbox(os.path.basename(lib_name),value=is_downloaded,disabled=True)
    if col2.button("Download",disabled=is_downloaded,key=lib_name):
        link = f"{RVC_DOWNLOAD_LINK}koboldcpp.exe"
        file_downloader((lib_name,link))
        st.experimental_rerun()

def render_model_checkboxes(generator):
    not_downloaded = []
    for model_path,link in generator:
        col1, col2, col3 = st.columns(3)
        is_downloaded = os.path.exists(model_path)
        col1.checkbox(os.path.basename(model_path),value=is_downloaded,disabled=True)
        if not is_downloaded: not_downloaded.append((model_path,link))
        col2.markdown(f"[Download Link]({link})")
        if col3.button("Download",disabled=is_downloaded,key=model_path):
            file_downloader((model_path,link))
            st.experimental_rerun()
    return not_downloaded

def rvc_index_path_mapper(params):
    (data_path, data) = params
    if "index" not in data_path.split(".")[-1]:
        return params
    else: return (os.path.join(BASE_MODELS_DIR,"RVC",".index",os.path.basename(data_path)), data) # index file

if __name__=="__main__":

    st.title("Download required models")

    with st.expander("Base Models"):
        generator = download_link_generator(RVC_DOWNLOAD_LINK, BASE_MODELS)
        to_download = render_model_checkboxes(generator)
        if st.button("Download All",key="download-all-base-models",disabled=len(to_download)==0):
            with ProgressBarContext(to_download,file_downloader,"Downloading models") as pb:
                pb.run()

    with st.container():
        if platform.system() == "Windows":
            render_download_ffmpeg()
            render_download_koboldcpp()
        elif platform.system() == "Linux":
            st.markdown("run `apt update && apt install -y -qq ffmpeg espeak` in your terminal")

    st.subheader("Required Models for inference")
    with st.expander("RVC Models"):
        file_uploader_form(
            os.path.join(BASE_MODELS_DIR,"RVC"),"Upload your RVC model",
            types=["pth","index","zip"],
            accept_multiple_files=True,
            params_mapper=rvc_index_path_mapper)
        generator = download_link_generator(RVC_DOWNLOAD_LINK, RVC_MODELS)
        to_download = render_model_checkboxes(generator)
        if st.button("Download All",key="download-all-rvc-models",disabled=len(to_download)==0):
            with ProgressBarContext(to_download,file_downloader,"Downloading models",parallel=True) as pb:
                pb.run()
    
    with st.expander("VITS Models"):
        generator = download_link_generator(RVC_DOWNLOAD_LINK, VITS_MODELS)
        to_download = render_model_checkboxes(generator)
        with ProgressBarContext(to_download,file_downloader,"Downloading models") as pb:
            st.button("Download All",key="download-all-vits-models",disabled=len(to_download)==0,on_click=pb.run)

    with st.expander("Chat Models"):
        col1, col2 = st.columns(2)
        stt_path = os.path.join(STT_MODELS_DIR,stt_checkpoint)
        is_downloaded = os.path.exists(stt_path)
        col1.checkbox(os.path.basename(stt_path),value=is_downloaded,disabled=True)
        if col2.button("Download",disabled=is_downloaded,key=stt_path):
            with st.spinner(f"Downloading {stt_checkpoint} to {stt_path}"):
                models = load_stt_models("speecht5") #hacks the from_pretrained downloader
                del models
                st.experimental_rerun()
        generator = [(os.path.join(BASE_MODELS_DIR,"LLM",os.path.basename(link)),link) for link in LLM_MODELS]
        to_download = render_model_checkboxes(generator)
        with ProgressBarContext(to_download,file_downloader,"Downloading models") as pb:
            st.button("Download All",key="download-all-chat-models",disabled=len(to_download)==0,on_click=pb.run)

    