# First
import os
import platform
import shutil
from types import FunctionType
import streamlit as st
from webui import MENU_ITEMS, get_cwd
st.set_page_config("RVC Chat",layout="centered",menu_items=MENU_ITEMS)

from webui.components import file_downloader, file_uploader_form

from webui.downloader import BASE_CACHE_DIR, BASE_MODELS, BASE_MODELS_DIR, GIT_REPOS, LLM_MODELS, RVC_DOWNLOAD_LINK, RVC_MODELS, SD_MODELS, download_link_generator, git_install

CWD = get_cwd()

from webui.contexts import ProgressBarContext

def render_download_lib(lib_name: str):
    col1, col2 = st.columns(2)
    is_downloaded = os.path.exists(lib_name)
    col1.checkbox(os.path.basename(lib_name),value=is_downloaded,disabled=True)
    if col2.button("Download",disabled=is_downloaded,key=lib_name):
        link = f"{RVC_DOWNLOAD_LINK}{lib_name}"
        file_downloader((lib_name,link))
        st.experimental_rerun()

def render_install_git(call_back: FunctionType=None):
    col1, col2 = st.columns(2)
    for url,location in GIT_REPOS:
        lib_name = os.path.basename(url)
        is_downloaded = os.path.exists(location)
        col1.checkbox(lib_name,value=is_downloaded,disabled=True)
        if col2.button("Install",disabled=is_downloaded,key=lib_name):
            git_install(url, location)
            if call_back: call_back(lib_name, location)
            st.experimental_rerun()

def after_git(lib_name, location):
    if "ComfyUI" in lib_name:
        if os.path.exists(location):
            shutil.copy("extra_model_paths.yaml",location) # copy configs
            # install image tagger
            url = "https://github.com/pythongosssss/ComfyUI-WD14-Tagger"
            dir = os.path.join(location,"custom_nodes","ComfyUI-WD14-Tagger")
            git_install(url, dir)
            wd14_tagger_model = os.path.join(dir,"models","wd-v1-4-convnextv2-tagger-v2.onnx")
            if not os.path.isfile(wd14_tagger_model):
                file_downloader((wd14_tagger_model,"https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2/resolve/main/model.onnx"))

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
            render_download_lib("ffmpeg.exe")
            render_download_lib("koboldcpp.exe")
        elif platform.system() == "Linux":
            st.markdown("run `apt update && apt install -y -qq ffmpeg espeak` in your terminal")
        render_install_git(after_git)

    st.subheader("Required Models for inference")
    with st.expander("Voice Models"):
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
    
    with st.expander("Chat Models"):
        generator = [(os.path.join(BASE_MODELS_DIR,"LLM",os.path.basename(link)),link) for link in LLM_MODELS]
        to_download = render_model_checkboxes(generator)
        with ProgressBarContext(to_download,file_downloader,"Downloading models") as pb:
            st.button("Download All",key="download-all-chat-models",disabled=len(to_download)==0,on_click=pb.run)

    with st.expander("Image Models"):
        generator = [(os.path.join(BASE_MODELS_DIR,"SD",os.path.basename(link)),link) for link in SD_MODELS]
        to_download = render_model_checkboxes(generator)
        with ProgressBarContext(to_download,file_downloader,"Downloading models") as pb:
            st.button("Download All",key="download-all-sd-models",disabled=len(to_download)==0,on_click=pb.run)

    