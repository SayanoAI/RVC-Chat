from functools import lru_cache
import json
import os
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from webui.chat import Character
from webui.downloader import BASE_MODELS_DIR
from webui.image_generation import generate_prompt
from webui import ObjectNamespace

FUNCTION_LIST = [
    ObjectNamespace(
        description = "can you draw an [object]?|show me what you look like|send me a picture of [object]|draw an [object] for me|please redraw the image [with/without these conditions]|POSITIVE: [string of words to include in the image]\n\nNEGATIVE: [string of words to remove from the image]",
        function = "generate_prompt",
        arguments = ["positive","positive_suffix","negative","orientation","seed","steps","cfg","randomize","checkpoint"],
        positive = "string containing everything the user wants to include in the image (e.g. animals, accessories, actions, etc.)",
        positive_suffix = "string containing the subject's physical appearance (e.g. hair color, eye color, clothes, etc.)",
        negative = "string containing anything the user wants to fix or remove from the drawing (e.g. extra fingers, missing limbs, errors, etc.)",
        orientation = "string describing the orientation of the image [square,portrait,landscape]",
        seed = "number to initialize the image with",
        steps = "number of steps to sample (20-40)",
        cfg = "number to show how closely to follow the prompt (7.0-12.0)",
        randomize="boolean to randomize the seed based on what the user wants (create new drawing=true, modify existing drawing=false)",
        checkpoint="string for the default checkpoint",
        STOP="boolean to stop the function call (set to 'True' if the assistant is unhappy)"
    ),
]
FUNCTION_MAP = ObjectNamespace(
    generate_prompt=generate_prompt
)

def get_function(char, query: str, threshold=1.,verbose=False):
    results = char.ltm.get_query(query=query,include=["metadatas", "distances"],type="function",threshold=threshold, verbose=verbose)
    if len(results): # found function
        metadata = results[0]["metadata"]
        return metadata
    else:
        print("failed to find function")
        return None

def get_args(char: "Character", arguments: str, template: str, prompt: str, reply: str, use_grammar=False):
    grammar = load_json_grammar() if use_grammar else ""
    
    try:
        chat_history = char.build_context(prompt)
        prompt_template = char.model_data["config"]["prompt_template"].format(
            instruction=f"Construct a JSON object with the following {arguments} keys, based on ({template}), using the following CHAT_HISTORY and REPLY:",
            context=f"CHAT_HISTORY: {chat_history}\n\n{char.name}'s REPLY: {reply}",
            prompt=f"Use nouns and verbs from the CHAT_HISTORY and REPLY to construct the JSON object based on the user's request: {prompt}."
        )
        
        generator = char.LLM(
            prompt_template,
            stop=char.model_data["config"]["stop_words"].split(",")+["\n\n"],
            grammar=grammar,
            stream=True,
            max_tokens=1024,
            mirostat_mode = 2,
            mirostat_tau = 4.0,
            mirostat_eta = 0.2,
        )
        for response in generator: pass
        args = json.loads(response["choices"][0]["text"])
        return args
    except Exception as e:
        print(f"failed to parse arguments: {e}")
        
    return None

@lru_cache
def load_json_grammar(fname=os.path.join(BASE_MODELS_DIR,"LLM","json.gbnf")):
    with open(fname,"r") as f:
        grammar = f.read()

    print(f"{grammar=}")
    return grammar

def call_function(character, prompt: str, context: str, threshold=1., retries=3, use_grammar=False, verbose=False):
    try:
        metadata = get_function(character, prompt, threshold, verbose)
        if metadata is None: metadata = get_function(character, context, threshold, verbose)
        if metadata and metadata["function"] in FUNCTION_MAP:
            while retries>0:
                args = get_args(character, metadata['arguments'], metadata['template'], prompt, context, use_grammar=use_grammar)
                
                if args:
                    try:
                        is_stopped = args.get("STOP",False)
                        if is_stopped and int(is_stopped):
                            print("character refused to execute function")
                            return False
                        else:
                            args = {k:args[k] for k in args if k in metadata['arguments']}
                            return metadata['function'], args, FUNCTION_MAP[metadata["function"]](**args)
                    except Exception as e:
                        print(e)
                        
                retries-=1
    except Exception as e:
        print(e)

    return None

def load_functions(vdb):
    for data in FUNCTION_LIST:
        print(f"{data=}")
        descriptions = data["description"].split("|")
        for description in descriptions:
            args = dict(data)
            args.update(description=description)
            print(f"{description=}\n")
            vdb.add_function(**args)