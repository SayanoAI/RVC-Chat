from functools import lru_cache
import json
import os
import re
from typing import List
from webui.downloader import BASE_MODELS_DIR
from webui.image_generation import generate_prompt
from webui import ObjectNamespace

FUNCTION_LIST = [
    ObjectNamespace(
        description = "can you draw me a [object]|show me what you look like|send me a picture of [object]|draw a [image] for me|POSITIVE: [keywords]\n\nNEGATIVE: [keywords]",
        function = "generate_prompt",
        arguments = ["positive","negative"],
        positive = "a list of comma separated words that describes the drawing",
        negative = "a list of comma separated words to avoid in the drawing"
    ),
]
FUNCTION_MAP = ObjectNamespace(
    generate_prompt=generate_prompt
)

def get_function(char, query: str, threshold=1.):
    results = char.ltm.get_query(query=query,include=["metadatas", "distances"],type="function",threshold=threshold)
    if len(results): # found function
        metadata = results[0]["metadata"]
        return metadata
    else: return None

def get_args(char, arguments: str, template: str, prompt: str, context: str, use_grammar=False):
    grammar = load_json_grammar() if use_grammar else ""
    print(f"{grammar=}")
    
    try:
        prompt_template = char.model_data["config"]["prompt_template"].format(
            instruction=f"Provide {arguments} as a JSON object based on this template ({template}) using the following data:",
            context=context,
            prompt=prompt
        )
        print(f"{prompt_template=}")
        generator = char.LLM(
            prompt_template,
            stop=char.model_data["config"]["stop_words"].split(",")+["\n\n"],
            grammar=grammar,
            stream=True,
            max_tokens=256,
            mirostat_mode = 2,
            mirostat_tau = 4.0,
            mirostat_eta = 0.2,
        )
        for response in generator: pass
        args = json.loads(response["choices"][0]["text"])
        if args: return {k:args[k] for k in args if k in arguments}
    except Exception as e:
        print(f"failed to parse arguments: {e}")
        
    return None

@lru_cache
def load_json_grammar(fname=os.path.join(BASE_MODELS_DIR,"LLM","json.gbnf")):
    with open(fname,"r") as f:
        grammar = f.read()
    return grammar

def call_function(character, prompt: str, context: str, threshold=1., retries=3, use_grammar=False):
    try:
        metadata = get_function(character, prompt, threshold)
        if metadata and metadata["function"] in FUNCTION_MAP:
            while retries>0:
                args = get_args(character, metadata['arguments'], metadata['template'], prompt, context, use_grammar=use_grammar)
                print(f"{args=}")
                if args: return FUNCTION_MAP[metadata["function"]](**args)
                retries-=1
    except Exception as e:
        print(e)

    return None

def load_functions(vdb):
    for data in FUNCTION_LIST:
        descriptions = data["description"].split("|")
        for description in descriptions:
            print(description)
            data["description"] = description
            vdb.add_function(**data)