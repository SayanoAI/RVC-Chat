import json
import os
import random
from webui.downloader import BASE_MODELS_DIR
from webui.kobold_cpp import Llama
from webui.utils import ObjectNamespace
from webui.chat import Character
import sounddevice as sd

def chat(char,prompt="Continue the story without me"):
    print(f"{char.user}: {prompt}")
    full_response = ""
    for response in char.generate_text(prompt):
        full_response += response
    audio = char.text_to_speech(full_response)
    if audio: sd.play(*audio)            
    char.messages.append({"role": char.user, "content": prompt}) #add user prompt to history
    char.messages.append({
        "role": char.name,
        "content": full_response,
        "audio": audio
    })
    print(f"{char.name}: {full_response}")
    return full_response

def get_function(char, query, threshold=1.):
    results = char.ltm.get_query(query=query,include=["metadatas", "distances"],type="function",threshold=threshold)
    if len(results): # found function
        metadata = results[0]["metadata"]
        return metadata
    else: return None

def get_args(char, arguments, template, prompt, retries=3):
    grammar = load_json_grammar()
    response = char.LLM(f"""
              <|im_start|>system
              Complete these arguments ({arguments}) using this template ({template}) while following the user's request.<|im_end|>
              <|im_start|>user
              {prompt}<|im_end|>
              """,grammar=grammar,stop=["<|im_start|>","<|im_end|>"])
    try:
        args = json.loads(response["choices"][0]["text"])
        if args: args = {k:args[k] for k in args if k in arguments}
    except Exception as e:
        print(e)
        args = get_args(char, arguments, template, prompt, retries-1) if retries>0 else None
    return args   

def load_json_grammar(fname=os.path.join(BASE_MODELS_DIR,"LLM","json.gbnf")):
    with open(fname,"r") as f:
        grammar = f.read()
    return grammar

functions = [
    ObjectNamespace(
        description = "Can you please play [song_name] for me?|I would love to hear you sing song_name.|Can you sing [song_name] for me?|Play [song_name], please.",
        function = "play_song",
        arguments = ["name","search"],
        name = {"type": "string", "description": "name of the song"},
        search = {"type": "boolean", "description": "search for the song if it doesn't exist"},
    ),
    ObjectNamespace(
        description = "Get the current weather in a [location]|How's the weather in a [location] in [temperature_unit]?|What's the weather like in [location]?",
        function = "get_current_weather",
        arguments = ["location","unit"],
        location = {"type": "string", "description": "location to check"},
        unit = {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "temperature unit"},
    ),
]

# dummy functions
def play_song(name,search=False):
    if search: print(f"Searching song on youtube")
    return f"playing {name}."

def get_current_weather(location, unit="celsius"):
    """Get the current weather in a given location"""

    temperature = random.randint(10,30)
    if unit=="fahrenheit": temperature*=3
    forecast = random.choice(["sunny", "windy"])
    
    return f"The temperature in {location} is {temperature} degrees {unit} and {forecast}"

def main():
    character = Character(
        character_file="./models/Characters/Amaryllis.json",
        model_file="./models/LLM/mistral-7b-openorca.Q4_K_M.gguf",
        user="Player"
    )
    character.LLM = Llama(
        character.model_file,
        n_ctx=character.model_data["params"]["n_ctx"],
        n_gpu_layers=character.model_data["params"]["n_gpu_layers"],
        verbose=True
    )
    character.loaded=True
    character
    character.ltm.clear()
    for data in functions:
        character.ltm.add_function(**data)
        
    prompt = "Can you sing despacito for me?"
    metadata = get_function(character,prompt,threshold=1.)
    print(f"{metadata=}")
    if metadata:
        args = get_args(character, metadata['arguments'], metadata['template'], prompt)
        print(f"{args=}")
        if args: print(eval(metadata["function"])(**args))
    prompt = "How's the weather in Canada in fahrenheit?"
    metadata = get_function(character,prompt,threshold=1.5)
    print(f"{metadata=}")
    if metadata:
        args = get_args(character, metadata['arguments'], metadata['template'], prompt)
        print(f"{args=}")
        if args: print(eval(metadata["function"])(**args))