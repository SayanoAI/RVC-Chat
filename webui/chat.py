from datetime import datetime
import io
import json
import os
import threading
from webui.utils import ObjectNamespace
from webui.kobold_cpp import Llama
import numpy as np
from lib.model_utils import get_hash
from tts_cli import generate_speech, load_stt_models, transcribe_speech
from webui.audio import bytes_to_audio, load_input_audio, remix_audio
from webui.downloader import BASE_MODELS_DIR, OUTPUT_DIR
import sounddevice as sd
from webui.utils import gc_collect
from webui.vector_db import VectorDB
from . import config
from vc_infer_pipeline import get_vc, vc_single
from pydub import AudioSegment

def init_model_params(): return ObjectNamespace(
        fname=None,n_ctx=2048,n_gpu_layers=0
    )
def init_model_config(): return ObjectNamespace(
        prompt_template = "",
        chat_template = "",
        instruction = "",
        mapper={
            "CHARACTER": "",
            "USER": ""
        },
        stop_words = ""
    )
def init_llm_options(): return ObjectNamespace(
        top_k = 42,
        repeat_penalty = 1.1,
        frequency_penalty = 0.,
        presence_penalty = 0.,
        mirostat_mode = 2,
        mirostat_tau = 4.0,
        mirostat_eta = 0.2,
        max_tokens = 1024,
        temperature = .8,
        top_p = .9,
    )
def init_model_data(): return ObjectNamespace(
        params = init_model_params(),
        config = init_model_config(),
        options = init_llm_options()
    )
def init_assistant_template(): return ObjectNamespace(
        background = "",
        personality = "",
        appearance = "",
        scenario = "",
        examples = [{"role": "", "content": ""}],
        greeting = "",
        name = ""
    )
def init_tts_options(): return ObjectNamespace(
        f0_up_key=0,
        f0_method=["rmvpe"],
        f0_autotune=False,
        merge_type="median",
        index_rate=.75,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=.2,
        protect=0.2,
        )
def init_character_data():
    return ObjectNamespace(
        assistant_template = init_assistant_template(),
        tts_options = init_tts_options(),
    )

def is_chub_card(data): return all([key in data for key in ["first_mes", "mes_example","description"]])
def import_chub_card(data):
    def parse_example(mes_example):
        examples = []
        for ex in mes_example.replace("<START>","").split("\n"):
            msg = ex.strip().split(":")
            if len(msg)<2: continue
            role = msg[0]
            content = " ".join(msg[1:])
            if len(content)>1: examples.append({
                "role": "USER" if role=="{{user}}" else "CHARACTER",
                "content": content.replace("{{char}}","{char}").replace("{{user}}","{user}")
            })
        return examples

    new_data = init_character_data()
    new_data["assistant_template"].update(ObjectNamespace(
        background = data["description"].replace("{{char}}","{char}").replace("{{user}}","{user}"),
        personality = data["personality"].replace("{{char}}","{char}").replace("{{user}}","{user}"),
        scenario = data["scenario"].replace("{{char}}","{char}").replace("{{user}}","{user}"),
        greeting = data["first_mes"].replace("{{char}}","{char}").replace("{{user}}","{user}"),
        name = data["name"],
        examples = parse_example(data["mes_example"])
    ))
    return new_data

def load_character_data(fname):
    
    with open(fname,"r") as f:
        data = json.load(f)
        loaded_state = ObjectNamespace(**data)
        
    if is_chub_card(loaded_state): return import_chub_card(loaded_state)
    else:
        new_data = init_character_data()
        new_data.update(loaded_state)
        new_data.assistant_template.update(loaded_state["assistant_template"])
        new_data.tts_options.update(loaded_state["tts_options"])
        return new_data

def load_model_data(model_file):
    fname = os.path.join(BASE_MODELS_DIR,"LLM","config.json")
    key = get_hash(model_file)
    model_data = init_model_data()

    with open(fname,"r") as f:
        data = ObjectNamespace(**json.load(f)) if os.path.isfile(fname) else model_data
        if key in data:
            data = data[key]
            if "version" in data and data["version"]==2:
                model_data["params"].update(data["params"])
                model_data["config"].update(data["config"])
                model_data["options"].update(data["options"])
            else:
                model_data["config"].update(
                    prompt_template = data["prompt_template"],
                    chat_template = data["chat_template"],
                    instruction = data["instruction"],
                    mapper = data["mapper"]
                )
                model_data["params"].update(
                    n_ctx=data["n_ctx"],
                    n_gpu_layers=data["n_gpu_layers"],
                    fname=data["name"]
                )
                model_data["options"].update(max_tokens = model_data["max_tokens"])

    return model_data

# Define a Character class
class Character:
    # Initialize the character with a name and a voice
    def __init__(self, character_file, model_file, memory = 0, user="",stt_method="speecht5",device=None,**kwargs):
        self.character_file = character_file
        self.model_file = model_file
        self.voice_model = None
        self.stt_models = None
        self.loaded = False
        self.sample_rate = 16000 # same rate as hubert model
        self.messages = []
        self.user = user
        self.is_recording = False
        self.stt_method = stt_method
        self.device=device
        self.autoplay = False
        self.recognizer = None
        self.listener = None
        self.lock = threading.Lock()
        self.has_voice = False
        self.ltm = VectorDB(character_file)

        #load data
        self.model_data = self.load_model(self.model_file)
        self.character_data = self.load_character(self.character_file)
        self.name = self.character_data["assistant_template"]["name"]

        # build context
        self.context_summary = ""
        self.context_size = 0
        self.context_index = 0
        self.memory = memory if memory else int(np.log(self.model_data["params"]["n_ctx"])*2.5)+1 # memory is log(context_window)*2.5 words/token
        self.max_memory = int(np.sqrt(self.model_data["params"]["n_ctx"]))+1
        self.context = self.build_context("")
        
    
    def load_character(self,fname):
        self.character_data = load_character_data(fname)
        self.name = self.character_data["assistant_template"]["name"]
        if fname!=self.character_file:
            self.clear_chat()
            # self.update_ltm(self.character_data["assistant_template"]["examples"])

        self.character_file = fname

        # load voice model
        try:
            if self.voice_model: del self.voice_model
            self.voice_model = get_vc(self.character_data["voice"],config=config,device=self.device)
            self.has_voice=True
        except Exception as e:
            print(f"failed to load voice {e}")
            self.has_voice=False
        
        return self.character_data

    def update_ltm(self,messages):
        model_config = self.model_data["config"]

        for ex in messages:
            if ex["role"] and ex["content"]:
                document =  model_config["chat_template"].format(role=model_config["mapper"][ex["role"]],content=ex["content"])
                self.ltm.add_document(document,
                    role=model_config["mapper"][ex["role"]],
                    content=ex["content"]
                )

    def load_model(self,fname):
        self.model_data = load_model_data(fname)
        self.model_file = fname
        return self.model_data

    def __del__(self):
        del self.ltm, self.messages
        self.unload()

    def __str__(self):
        return f"Character({self.character_file=},{self.model_file=},{self.loaded=})"

    def __missing__(self,attribute):
        return None

    def stop_listening(self):
        if self.listener:
            self.is_recording = False
            self.listener.join(1)

    def load(self,verbose=False):
        assert not self.loaded, "Model is already loaded"

        try:
            self.LLM = Llama(self.model_file,n_ctx=self.model_data["params"]["n_ctx"],n_gpu_layers=self.model_data["params"]["n_gpu_layers"],verbose=verbose)
            self.context_size = self.LLM.token_count(self.context)
            self.free_tokens = self.model_data["params"]["n_ctx"] - self.context_size

            if len(self.messages)==0 and self.character_data["assistant_template"]["greeting"] and self.user:
                self.messages.append(self.greeting_message)
            
            self.loaded=True
            return "Successfully loaded character!"
        except Exception as e:
            return f"Failed to load character: {e}"

    def unload(self):
        if self.LLM: del self.LLM
        if self.voice_model: del self.voice_model
        if self.stt_models: del self.stt_models
        if self.recognizer: del self.recognizer
        self.LLM = self.voice_model = self.stt_models = self.recognizer = None
        gc_collect()
        self.loaded=False
        self.is_recording = False
        self.stop_listening()
        print("models unloaded")

    def toggle_autoplay(self):
        self.autoplay = not self.autoplay
        self.is_recording = False

    def clear_chat(self):
        del self.messages
        self.ltm.clear()
        gc_collect()
        self.messages = [self.greeting_message]

    @property
    def save_dir(self):
        history_dir = os.path.join(OUTPUT_DIR,"chat",self.name)
        num = len(os.listdir(history_dir)) if os.path.exists(history_dir) else 0
        save_dir = os.path.join(history_dir,f"{datetime.now().strftime('%Y-%m-%d')}_chat{num}")
        return save_dir
    
    @property
    def greeting_message(self):
        greeting_message = {
            "role": self.character_data["assistant_template"]["name"],
            "content": self.character_data["assistant_template"]["greeting"].format(
                char=self.character_data["assistant_template"]["name"], user=self.user
            )}
        if self.has_voice:
            output_audio = self.text_to_speech(greeting_message["content"])
            if (output_audio):
                sd.play(*output_audio)
                greeting_message["audio"] = output_audio
        return greeting_message

    def save_history(self, save_dir = None, format="wav"):
        if not save_dir: save_dir = self.save_dir
        os.makedirs(save_dir,exist_ok=True)
        def role_mapper(role):
            if role==self.name: return "CHARACTER"
            elif role==self.user: return "USER"
            return role
        
        try:
            combined_audio = AudioSegment.empty()
            start_timestamp = 0

            # Create a list to store timestamps
            messages = []

            # Iterate through the list of dicts
            for data_dict in self.messages:
                # Extract audio file name and role from the dictionary
                audio = data_dict.get("audio")
                role = role_mapper(data_dict.get("role","")) if role_mapper else data_dict.get("role","")
                content = data_dict.get("content","").replace(self.user,"USER").replace(self.name,"CHARACTER")
                message = {"role": role,"content": content}

                if audio is not None:
                    audio_data, sr = remix_audio(audio,to_int16=format=="wav",to_mono=True)
                    print([audio_data.dtype,audio_data.dtype.itemsize,sr,audio_data.max()])

                    # Load the audio file using pydub
                    audio_segment = AudioSegment(audio_data.tobytes(), sample_width=audio_data.dtype.itemsize, frame_rate=sr, channels=1)

                    # Append the audio segment to the combined audio
                    if len(combined_audio): combined_audio += audio_segment
                    else: combined_audio = audio_segment

                    # Calculate the timestamp for the end of this segment
                    end_timestamp = len(combined_audio)
                
                    message["timestamp"] = [start_timestamp, end_timestamp]
                    start_timestamp = end_timestamp
                
                messages.append(message)
            output_audio_filename = os.path.join(save_dir,f"messages.{format}")
            combined_audio.export(output_audio_filename, format=format)
            text = json.dumps({
                "messages":messages,
                "user":self.user,
                "context_index": self.context_index,
                "context_summary": self.context_summary
            },indent=2)
            with open(os.path.join(save_dir,"messages.json"),"w") as f:
                f.write(text)
            return f"Chat successfully saved in {save_dir}"
        except Exception as e:
            return f"Chat failed to save: {e}"

    def load_history(self,history_file):
        
        messages = []
        save_dir = os.path.dirname(history_file)

        try:
            with open(history_file,"r") as f:
                data = json.load(f)
                saved_messages = data.get("messages",[])
                self.user = data.get("user",self.user)
                self.context_index = data.get("context_index",self.context_index)
                self.context_summary = data.get("context_summary",self.context_summary)
                
            combined_audio_file = os.path.join(save_dir,"messages")
            if os.path.isfile(f"{combined_audio_file}.wav"): combined_audio = AudioSegment.from_wav(f"{combined_audio_file}.wav")
            elif os.path.isfile(f"{combined_audio_file}.mp3"): combined_audio = AudioSegment.from_mp3(f"{combined_audio_file}.mp3")
            else: combined_audio=None

            for msg in saved_messages:
                if msg["role"]=="CHARACTER": role = self.name
                elif msg["role"]=="USER": role = self.user
                else: role = msg["role"]
                content = msg['content'].replace("USER",self.user).replace("CHARACTER",self.name)
                message = {
                    "role": role,
                    "content": content
                }
                if "audio" in msg: #legacy loading
                    fname=os.path.join(save_dir,msg["audio"])
                    message["audio"] = load_input_audio(fname)
                elif "timestamp" in msg and combined_audio: #v2
                    start_timestamp,end_timestamp = msg["timestamp"]
                    segment = combined_audio[start_timestamp:end_timestamp]
                    with io.BytesIO() as temp_wav_file:
                        segment.export(temp_wav_file, format="wav")
                        message["audio"] = bytes_to_audio(temp_wav_file)
                messages.append(message)
            self.messages = messages
            # summarize messages after load
            self.context_index = 0
            self.context = self.build_context("")
            
            return f"Chat successfully loaded from {save_dir}!"
        except Exception as e:
            return f"Chat failed to load: {e}"

    # Define a method to generate text using llamacpp model
    def generate_text(self, input_text):
        assert self.loaded, "Please load the models first"

        model_config = self.model_data["config"]
        # Send the input text to llamacpp model as a prompt
        self.context = self.build_context(input_text)
        generator = self.LLM.create_completion(
            self.context,stream=True,stop=["USER","CHARACTER"]+model_config["stop_words"].split(","),
            **self.model_data["options"])
        
        for completion_chunk in generator:
            response = completion_chunk['choices'][0]['text']
            yield response

    def chat_mapper(self,role):
        model_config = self.model_data["config"]
        assistant_template = self.character_data["assistant_template"]
        if role==self.user: return model_config["mapper"]["USER"]
        elif role==assistant_template["name"]: return model_config["mapper"]["CHARACTER"]
        else: return role

    def build_context(self,prompt: str):
        model_config = self.model_data["config"]
        assistant_template = self.character_data["assistant_template"]

        # clear chat history if memory maxed
        if len(self.messages[self.context_index:])>self.max_memory:
            print(f"moving memory to ltm: {self.context_index}")
            self.update_ltm(self.messages[:self.context_index])
            self.context_index+=self.max_memory
            self.context_summary = self.summarize_context()
            
        # summarize memory
        elif self.loaded and self.LLM.token_count(self.context)+self.context_size>self.model_data["params"]["n_ctx"]:
            self.context_index+=self.memory
            self.context_summary = self.summarize_context() #summarizes the past
        
        # Concatenate chat history and system template
        history = self.ltm.get_query(prompt,n_results=self.memory)
        if len(history): print(history)

        messages = "\n".join([
            model_config["chat_template"].format(role=self.chat_mapper(ex["role"]),content=ex["content"])
            for ex in self.messages[self.context_index:]
        ])
        history = "\n".join([
            model_config["chat_template"].format(role=self.chat_mapper(ex["metadata"]["role"]),content=ex["metadata"]["content"])
            for ex in history
        ])
        examples = "\n".join([
            model_config["chat_template"].format(role=self.chat_mapper(ex["role"]),content=ex["content"])
            for ex in assistant_template["examples"]
        ])
            
        persona = "\n".join(
            f"{key.upper()}: {assistant_template[key]}"
            for key in ["background","personality","appearance","scenario"]
            if assistant_template[key]
        )

        context = "\n\n".join([f"PERSONA\n{persona}",f"EXAMPLES\n{examples}",f"HISTORY\n{history}",messages]).format(
            char=assistant_template["name"],
            user=self.user
        )

        instruction = model_config["instruction"].format(
            char=assistant_template["name"],
            user=self.user
        )
        
        chat_history_with_template = model_config["prompt_template"].format(
            context=context,
            instruction=instruction,
            char=assistant_template["name"],
            user=self.user,
            prompt=prompt
            )

        return chat_history_with_template
    
    def summarize_context(self):
        from webui.sumy_summarizer import get_summary

        history = "\n".join([self.context_summary] + [
            "{role}: {content}".format(role=self.chat_mapper(ex["role"]),content=ex["content"])
            for ex in self.messages[-self.memory:]
        ])
        num = int(np.sqrt(self.memory))+1

        completion = get_summary(history,num_sentences=num)
        
        print(completion)
        return completion

    # Define a method to convert text to speech
    def text_to_speech(self, text):
        if self.voice_model is None: self.voice_model = get_vc(self.character_data["voice"],config=config,device=self.device)
        tts_audio = generate_speech(text,method=self.character_data["tts_method"], speaker=self.name, device=config.device, dialog_only=True)
        output_audio = vc_single(input_audio=tts_audio,**self.voice_model,**self.character_data["tts_options"])
        return output_audio

    # Define a method to run the STT and TTS in the background and be non-blocking
    def speak_and_listen(self):
        assert self.loaded, "Please load the models first"
        import speech_recognition as sr
        
        # Create a speech recognizer instance
        self.stt_models = load_stt_models(self.stt_method) #speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 2000
        self.recognizer.pause_threshold = 2.
        self.is_recording = True
        
        # Start listening to the microphone in the background and call the callback function when speech is detected
        self.autoplay = False
        
        # Create a microphone instance
        # Start listening to mic
        self.listener = threading.Thread(target=self.microphone_callback,daemon=True,name="listener")
        self.listener.start()
        # self.stop_listening = self.recognizer.listen_in_background(sr.Microphone(sample_rate=self.sample_rate), self.microphone_callback)
        

    # Define a callback function that will be called when speech is detected
    def microphone_callback(self):
        import speech_recognition as sr
        with sr.Microphone(sample_rate=self.sample_rate) as source, self.lock:
            self.recognizer.adjust_for_ambient_noise(source, duration=self.recognizer.pause_threshold)

        while self.is_recording:
            with sr.Microphone(sample_rate=self.sample_rate) as source, self.lock:
                try:
                    audio = self.recognizer.listen(source)

                    if not self.is_recording:
                        return self.stop_listening()
                    
                    print("listening to mic...")
                    
                    prompt, input_audio = transcribe_speech(audio,stt_models=self.stt_models,stt_method=self.stt_method,denoise=True)
                    if prompt is not None and len(prompt)>1:
                        print(f"{self.name} heard: {prompt}")
                        full_response = ""
                        for response in self.generate_text(prompt):
                            full_response += response

                        if self.has_voice:
                            output_audio = self.text_to_speech(full_response)
                            if output_audio: sd.play(*output_audio)
                        else:
                            output_audio = None
                        
                        self.messages.append({"role": self.user, "content": prompt, "audio": input_audio}) #add user prompt to history
                        self.messages.append({
                            "role": self.name,
                            "content": full_response,
                            "audio": output_audio
                            })
                        print(f"{self.name} said: {full_response}")
                        sd.wait() # wait for audio to stop playing
                except Exception as e:
                    print(e)

class Recorder:
    # Initialize the character with a name and a voice
    def __init__(self, stt_method="speecht5", device=None):
        self.voice_model = None
        self.stt_models = None
        self.loaded = False
        self.sample_rate = 16000 # same rate as hubert model
        self.is_recording = False
        self.stt_method = stt_method
        self.device=device
        self.autoplay = False
        self.recognizer = None
        self.listener = None
        self.lock = threading.Lock()
        self.text=None


    def __del__(self):
        self.stop_listening()
        gc_collect()
        print("models unloaded")

    # Define a method to run the STT and TTS in the background and be non-blocking
    def start_listening(self):
        import speech_recognition as sr
        
        # Create a speech recognizer instance
        self.stt_models = load_stt_models(self.stt_method) #speech recognition
        self.recognizer = sr.Recognizer()
        # self.recognizer.energy_threshold = 2000
        self.recognizer.pause_threshold = 2.
        self.is_recording = True
        
        # Start listening to the microphone in the background and call the callback function when speech is detected
        self.autoplay = False
        
        # Create a microphone instance
        # Start listening to mic
        self.listener = threading.Thread(target=self.microphone_callback,daemon=True,name="listener")
        self.listener.start()
       
    # Define a callback function that will be called when speech is detected
    def microphone_callback(self):
        import speech_recognition as sr
        # with sr.Microphone(0,sample_rate=self.sample_rate) as source, self.lock:
        #     self.recognizer.adjust_for_ambient_noise(source, duration=self.recognizer.pause_threshold)

        while self.is_recording:
            with sr.Microphone(sample_rate=self.sample_rate) as source, self.lock:
                try:
                    sd.wait()
                    audio = self.recognizer.listen(source)

                    if not self.is_recording:
                        return self.stop_listening()
                    
                    print("listening to mic...")

                    # if len(audio.frame_data)>self.sample_rate*2:
                    input_audio = bytes_to_audio(audio.get_wav_data())
                    self.text = transcribe_speech(input_audio,stt_models=self.stt_models,stt_method=self.stt_method,denoise=True)
                    print(self.text)
                except Exception as e:
                    print(e)

    def stop_listening(self):
        if self.listener:
            print("stopped listening to mic...")
            self.is_recording = False
            self.listener.join(1)
            del self.listener, self.recognizer, self.stt_models
            self.listener = self.recognizer = self.stt_models = None