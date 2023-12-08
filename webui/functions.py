from functools import lru_cache
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webui.chat import Character
    from webui.vector_db import VectorDB
from webui.downloader import BASE_MODELS_DIR
from webui.image_generation import generate_prompt, modify_image
from webui import ObjectNamespace

FUNCTION_LIST = [
    ObjectNamespace(
        documents = "can you draw me a [object]?|show me what [noun/pronoun/action] looks like|send me a picture of [object]|draw an [object] for me|SUBJECT: [words to describe the main image subject]\nDESCRIPTION: [words to describe the main subject's appearance]\nENVIRONMENT: [words to describe the environment or items in it]\nNEGATIVE: [words to describe things to remove from the image]",
        function = "generate_prompt",
        arguments = ["subject","description","emotion","environment","negative","orientation","seed","steps","cfg","randomize","scale","style"],
        subject = "string containing main subject of the image (e.g. a playful cat, a beautiful woman, an old man, etc.)",
        description = "string describing the subject's physical appearance (e.g. hair color, eye color, clothes, etc.)",
        emotion = "string describing the overall emotion of the subject or environment (e.g. sombre, smile, happy, etc.)",
        environment = "string describing everything else you want to include in the image (e.g. animals, environment, background, etc.)",
        negative = "string containing anything the user wants to fix or remove from the drawing (e.g. extra fingers, missing limbs, errors, etc.)",
        orientation = "string describing the orientation of the image [square,portrait,landscape]",
        seed = "number to initialize the image with",
        steps = "number of steps to sample (20-40)",
        cfg = "number to show how closely to follow the prompt (7.0-12.0)",
        randomize="boolean to randomize the seed based on what the user wants (create new drawing=true, modify existing drawing=false)",
        scale="number to scale the image by (1.0-2.0)",
        style="string to describe the style of image (e.g. drawing, photograph, sketch, etc.)",
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    ObjectNamespace(
        documents = "please redraw your image [with/without these conditions]|can you fix this [conditions]?|there's [something wrong with the image]...|please change [the image] to [something else]|can you make [things to change in the image]?|can you draw [object with conditions]",
        function = "modify_image",
        arguments = ["subject","description","emotion","environment","negative","steps","cfg","style","changes"],
        subject = "string containing main subject of the image (e.g. a playful cat, a beautiful woman, an old man, etc.)",
        description = "string describing the subject's physical appearance (e.g. hair color, eye color, clothes, etc.)",
        emotion = "string describing the overall emotion of the subject or environment (e.g. sombre, smile, happy, etc.)",
        environment = "string describing everything else you want to include in the image (e.g. animals, environment, background, etc.)",
        negative = "string containing anything the user wants to fix or remove from the drawing (e.g. extra fingers, missing limbs, errors, etc.)",
        steps = "number of steps to sample (20-40)",
        cfg = "number to show how closely to follow the prompt (7.0-12.0)",
        style="string to describe the style of image (e.g. drawing, photograph, sketch, etc.)",
        change_ratio="string to describe how much to alter the image (small|medium|large)",
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
]
FUNCTION_MAP = ObjectNamespace(
    generate_prompt=generate_prompt,
    modify_image=modify_image
)

def get_function(char: "Character", query: str, threshold=1.,verbose=False):
    results = char.ltm.get_query(query=query,include=["metadatas", "distances"],type="function",threshold=threshold, verbose=verbose)
    if len(results): # found function
        metadata = results[0]["metadata"]
        return metadata
    else:
        print("failed to find function")
        return None

def get_args(char: "Character", instructions: str, prompt: str, use_grammar=False):
    grammar = load_json_grammar() if use_grammar else ""
    
    try:
        prompt_template = char.model_data["config"]["prompt_template"].format(
            system=instructions,
            history=char.summarized_history,
            prompt=prompt
        )
        
        generator = char.LLM(
            prompt_template,
            stop=char.model_data["config"]["stop_words"].split(",")+["\n\n\n"],
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

def call_function(character: "Character", prompt: str, reply: str, threshold=1., retries=3, use_grammar=False, verbose=False, **kwargs):
    try:
        errors = []
        metadata = get_function(character, prompt, threshold, verbose)
        if metadata is None: metadata = get_function(character, reply, threshold, verbose)
        if metadata and metadata["function"] in FUNCTION_MAP:
            instructions = metadata['instructions'].format(template=metadata['template'],context=reply)
            while retries>0:
                error_prompt = instructions+f"\nPlease avoid the following errors when generating your prompt: {', '.join(errors)}" if len(errors) else ""
                args = get_args(character, instructions=instructions, prompt=prompt+error_prompt, use_grammar=use_grammar)
                
                if args:
                    try:
                        args = {k:args[k] for k in args if k in metadata['arguments']}
                        if "image" in kwargs:
                            image = kwargs.pop("image")
                            args["image"] = character.get_image if image is None else image
                        args.update(kwargs)
                        results = FUNCTION_MAP[metadata['function']](**args)
                        if results is not None: return results
                    except Exception as e:
                        print(e)
                        errors.append(e)
                retries-=1
    except Exception as e:
        print(e)

    return None

def load_functions(vdb: "VectorDB"):
    for data in FUNCTION_LIST:
        print(f"{data=}")
        descriptions = data["documents"].split("|")
        for description in descriptions:
            args = dict(data)
            args.update(documents=description)
            print(f"{description=}\n")
            vdb.add_function(**args)