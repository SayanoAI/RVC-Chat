# First
import os
import platform
import shutil
from types import FunctionType
import streamlit as st
from webui import MENU_ITEMS, get_cwd
st.set_page_config("RVC Chat",layout="centered",menu_items=MENU_ITEMS)

from webui.components import file_downloader, file_uploader_form

from webui.downloader import BASE_MODELS, BASE_MODELS_DIR, GIT_REPOS, LLM_MODELS, RVC_DOWNLOAD_LINK, RVC_MODELS, SD_MODELS, download_link_generator, git_install, git_update

from webui.contexts import ProgressBarContext

get_cwd()

LLM_MAPPER = {
    "neuralhermes-2.5-mistral-7b.Q4_K_M.gguf": "8GB VRAM",
    "stablelm-zephyr-3b.Q5_K_M.gguf": "< 8GB VRAM",
    "LLaMA2-13B-Tiefighter.Q4_K_M.gguf": "> 8GB VRAM"
}

def render_download_lib(lib_name: str):
    col1, col2 = st.columns(2)
    is_downloaded = os.path.exists(lib_name)
    col1.checkbox(os.path.basename(lib_name),value=is_downloaded,disabled=True)
    if col2.button("Download",disabled=is_downloaded,key=lib_name):
        link = f"{RVC_DOWNLOAD_LINK}{lib_name}"
        file_downloader((lib_name,link))
        st.experimental_rerun()

def render_install_git(call_back: FunctionType=None):
    for url,dirname in GIT_REPOS:
        col1, col2 = st.columns(2)
        lib_name = os.path.basename(url)
        location = os.path.join(dirname,lib_name)
        is_downloaded = os.path.exists(location)
        col1.checkbox(lib_name,value=is_downloaded,disabled=True)
        if col2.button("Update" if is_downloaded else "Install",key=lib_name):
            if (git_update(location) if is_downloaded else git_install(url, location)):
                if call_back and call_back(lib_name, location):
                    st.toast(f"Successfully installed {lib_name} in {location}!")

def after_git(lib_name, location):
    if "ComfyUI-WD14-Tagger" in lib_name:
        if os.path.exists(location):
            # download image tagger model
            wd14_tagger_model = "wd-v1-4-moat-tagger-v2"
            for file in ["model.onnx","selected_tags.csv"]:
                wd14_tagger_path = os.path.join(location,"models",wd14_tagger_model + os.path.splitext(file)[-1])
                if not os.path.isfile(wd14_tagger_path):
                    file_downloader((wd14_tagger_path,f"https://huggingface.co/SmilingWolf/{wd14_tagger_model}/resolve/main/{file}"))

def render_model_checkboxes(generator,mapper={}):
    not_downloaded = []
    for model_path,link in generator:
        col1, col2, col3 = st.columns(3)
        is_downloaded = os.path.exists(model_path)
        name = os.path.basename(model_path)
        label = mapper.get(name)
        if label: name+=f" ({label})" # attach label
        col1.checkbox(name,value=is_downloaded,disabled=True)
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
        to_download = render_model_checkboxes(generator,mapper=LLM_MAPPER)
        with ProgressBarContext(to_download,file_downloader,"Downloading models") as pb:
            st.button("Download All",key="download-all-chat-models",disabled=len(to_download)==0,on_click=pb.run)

    with st.expander("Image Models"):
        generator = [(os.path.join(BASE_MODELS_DIR,"SD",os.path.basename(link)),link) for link in SD_MODELS]
        to_download = render_model_checkboxes(generator)
        with ProgressBarContext(to_download,file_downloader,"Downloading models") as pb:
            st.button("Download All",key="download-all-sd-models",disabled=len(to_download)==0,on_click=pb.run)

    