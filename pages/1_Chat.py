import io
import os
from PIL import Image
import streamlit as st
from webui import MENU_ITEMS, SERVERS, ObjectNamespace, get_cwd, i18n
from webui.downloader import OUTPUT_DIR, save_file
from webui.functions import call_function
from webui.image_generation import describe_image, generate_images
st.set_page_config(layout="wide",menu_items=MENU_ITEMS)

from webui.chat import Character
from webui.components import file_uploader_form
import sounddevice as sd
from webui.contexts import SessionStateContext
from webui.utils import get_filenames, get_index, get_optimal_torch_device, gc_collect

CWD = get_cwd()

def get_model_list():
    models_list =  [os.path.basename(path) for path in get_filenames(root=os.path.join(CWD,"models","SD"),exts=["safetensors","ckpt"])]
    return models_list

def get_voice_list():
    models_list = [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models","RVC"),exts=["pth"])]
    return models_list

def get_character_list():
    models_list =  [os.path.relpath(path,CWD) for path in get_filenames(root=os.path.join(CWD,"models","Characters"),exts=["json"])]
    return models_list

def init_state():
    state = ObjectNamespace(
        voice_models=get_voice_list(),
        characters=get_character_list(),
        model_list=get_model_list(),
        messages = [],
        user = "User",
        memory=10,
        threshold=1.5,
        character=None,
        checkpoint=None,
        device=get_optimal_torch_device(),
    )
    return state

def refresh_data(state):
    state.model_list = get_model_list()
    state.voice_models = get_voice_list()
    state.characters = get_character_list()
    gc_collect()
    return state

@st.cache_data
def format_label(x: str): return os.path.basename(x)

if __name__=="__main__":
    with SessionStateContext("chat",init_state()) as state:
        
        st.title("RVC Chat")

        hint = st.empty()

        if st.button("Refresh Files"):
            state = refresh_data(state)
        col1, col2, col3 = st.columns(3)
        
        state.user = col1.text_input("Your Name", value=state.user)
        state.selected_character = col2.selectbox("Your state.character",
                                              options=state.characters,
                                              index=get_index(state.characters,state.selected_character),
                                              format_func=format_label)
        state.checkpoint = col3.selectbox(
                i18n("Checkpoint"),
                options=state.model_list,
                index=get_index(state.model_list,state.checkpoint))
        
        state.threshold = col3.select_slider("Threshold to trigger function calls (0=never trigger, 2=always trigger)",options=[0,0.5,1,1.5,2],value=state.threshold)

        with col2:
            c1, c2 = st.columns(2)
            
            
            if c2.button("Load Character" if state.character else "Start Chatting",disabled=not (state.selected_character and state.user),type="primary"):
                with st.spinner("Loading model..."):
                    if state.character:
                        state.character.memory = state.memory
                        state.character.load_character(state.selected_character)
                        state.character.user = state.user
                        state.character.loaded = True
                        
                        # if hash(state.character.model_file)!=hash(state.selected_llm):
                        state.character.load_model(SERVERS.LLM_MODEL)
                        state.character.unload()
                        st.toast(state.character.load())
                    else:
                        state.character = Character(
                            character_file=state.get("selected_character"),
                            model_file=state.get("selected_llm"),
                            user=state.get("user"),
                            device=state.get("device"),
                            memory=state.memory
                        )
                        if state.character and not state.character.loaded: st.toast(state.character.load())

        chat_disabled = state.character is None or not state.character.loaded
        if chat_disabled: hint.warning("Enter your name, select your state.character, and choose an image generation model to get started!")

        if not chat_disabled:
            state.character.has_voice = col3.checkbox("Voiced", value=state.character.has_voice) # mutes character

            # save/load chat history
            save_dir = os.path.join(OUTPUT_DIR,"chat",state.character.name)
            file_uploader_form(save_dir,title="Upload your chat history",types=["zip"])
            state.history_file = st.selectbox("Continue a saved chat history",options=[""]+get_filenames(root=save_dir,name_filters=["json"]))
            col1,col2, col3 = st.columns(3)

            if col1.button("Save Chat",disabled=not state.character):
                save_dir = os.path.dirname(state.history_file) if state.history_file else None
                st.toast(state.character.save_history(save_dir))
                state = refresh_data(state)

            if col2.button("Load Chat",disabled=not state.history_file):
                st.toast(state.character.load_history(state.history_file))
                state.user = state.character.user

            if col3.button("Clear Chat",type="primary",disabled=state.character is None or len(state.character.messages)==0):
                state.character.clear_chat()

            # display chat messages
            for i,msg in enumerate(state.character.messages):
                with st.chat_message(msg["role"]):
                    audio = msg.get("audio")
                    image = msg.get("image")
                    st.write(msg["content"])
                    col1, col2, col3 = st.columns(3)

                    if image and len(image):
                        col1.image(image[0])
                        if col1.button("Save Image",key=f"Save{i}"):
                            img_name = os.path.join(OUTPUT_DIR,"chat",state.character.name,"images",f"{i}.png")
                            with io.BytesIO() as f:
                                image[0].save(f,format="PNG")
                                st.toast(f"Saved image: {save_file((img_name,f.getvalue()))}")

                    if audio and col2.button("Play",key=f"Play{i}"): sd.play(*msg["audio"])

                    if col3.button("Delete Message",key=f"Delete{i}"):
                        st.toast(f"Deleted message: {state.character.messages.pop(i)}")
                        st.experimental_rerun()

            action_placeholder = st.empty()
            prompt=st.chat_input(disabled=chat_disabled or state.character.autoplay)

            c1,c2,c3,c4,c5 = action_placeholder.columns(5)
            if c1.button("Summarize Context"):
                st.write(state.character.summarized_history)
            if c2.button("Toggle Autoplay",type="primary" if state.character.autoplay else "secondary" ):
                state.character.toggle_autoplay()
                st.experimental_rerun()
            if c3.button("Regenerate"):
                state.character.messages.pop()
                msg = state.character.messages.pop()
                # reuse prompt and last uploaded image
                prompt = msg["content"]
                if "image" in msg:
                    state.uploaded_image = msg["image"][0]
                    with io.BytesIO() as f:
                        state.uploaded_image.save(f,format="PNG")
                        state.tags = describe_image(f.getvalue())

            if c4.button("Clear Chat",key="clear-chat-2",type="primary"):
                state.character.clear_chat()
                st.experimental_rerun()

            with st.form("describe_image",clear_on_submit=True):
                img_file = st.file_uploader("Upload Image",type=["png","jpg","jpeg"])
                if st.form_submit_button(f"Show picture to {state.character.name}",type="primary"):
                    if img_file:
                        image = Image.open(img_file)
                        state.uploaded_image = image
                        with io.BytesIO() as f:
                            image.save(f,format="PNG")
                            state.tags = describe_image(f.getvalue())
            if state.uploaded_image: st.image(state.uploaded_image)
            if state.tags: st.write(f"Image caption: {state.tags}")
            
            if state.character.autoplay or prompt:
                state.character.is_recording=False
                if not state.character.autoplay:
                    st.chat_message(state.character.user).write(prompt)
                
                full_response = ""
                images = None
                with st.chat_message(state.character.name):
                    message_placeholder = st.empty()
                    if state.character.autoplay: augmented_prompt = "**Please continue the story without {{user}}'s input**" 
                    else:
                        augmented_prompt = f"{prompt}\nImage caption: {state.tags}" if state.tags else prompt
                    
                    with st.spinner(f"{state.character.name} is typing..."):
                        for response in state.character.generate_text(augmented_prompt):
                            full_response = response
                            message_placeholder.markdown(full_response)

                    if not state.character.autoplay:
                        message = {"role": state.character.user, "content": prompt}
                        if state.uploaded_image: message["image"] = [state.uploaded_image]
                        state.character.messages.append(message) #add user prompt to history
                        state.uploaded_image = None
                        state.tags = None

                    with st.spinner(f"{state.character.name} is thinking..."):
                        image_prompt = call_function(
                            state.character,
                            prompt=augmented_prompt,
                            reply=full_response,
                            use_grammar=True,
                            threshold=state.threshold,
                            verbose=True,
                            checkpoint=state.checkpoint,
                            positive_suffix=state.tags,
                            image=state.uploaded_image
                            ) # calls function
                        if image_prompt is not None:
                            with st.spinner(f"{state.character.name} is creating an image..."):
                                images = generate_images(image_prompt)
                                st.image(images)

                if state.character.has_voice:
                    audio = state.character.text_to_speech(full_response)
                    if audio:
                        if state.character.autoplay: sd.wait() # wait for speech to finish
                        sd.play(*audio)
                else:
                    audio = None
                
                state.character.messages.append({
                    "role": state.character.name,
                    "content": full_response,
                    "audio": audio,
                    "image": images
                    })
                
                gc_collect()
                st.experimental_rerun()

            
            
        