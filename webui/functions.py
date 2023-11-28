from functools import lru_cache
import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from webui.chat import Character
    from webui.vector_db import VectorDB
from webui.downloader import BASE_MODELS_DIR
from webui.image_generation import generate_prompt
from webui import ObjectNamespace

FUNCTION_LIST = [
    ObjectNamespace(
        documents = "Tool that enables the user to generate an image or picture based on a description",
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
        randomize="boolean to show whether to randomize the image (create new drawing=true, modify same drawing=false)",
        scale="number to scale the image by (1.0-2.0)",
        style="string to show the style of the image (anime|realistic)",
        STOP="boolean to stop the function call (set to 'True' if the assistant is unhappy)",
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    ObjectNamespace(
        documents = "Tool that enables the user to search youtube for songs",
        function = "search_youtube",
        arguments = ["query"],
        
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    ObjectNamespace(
        documents = "Tool that enables the user to download a youtube song given a url",
        function = "download_youtube",
        arguments = ["url"],
        
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    ObjectNamespace(
        documents = "Tool that enables the user to search for a song in the current directory",
        function = "search_song",
        arguments = ["song_name"],
        
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    ObjectNamespace(
        documents = "Tool that lets the user isolate the vocals from a song",
        function = "split_vocals",
        arguments = ["song_name"],
        
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    ObjectNamespace(
        documents = "Tool that enables the user to use AI to sing the song",
        function = "convert_vocals",
        arguments = ["voice_model","vocals"],
        
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    ObjectNamespace(
        documents = "Tool that enables the user to merge two audio data",
        function = "merge_audio",
        arguments = ["input_audio1","input_audio2"],
        
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
]
FUNCTION_MAP = ObjectNamespace(
    generate_prompt=generate_prompt,
    
)

def get_function(char, query: str, threshold=1.,verbose=False):
    # results = char.ltm.get_query(query=query,include=["metadatas", "distances"],type="function",threshold=threshold, verbose=verbose)
    functions = "\n".join([f"{data.function}({','.join(data.arguments)}): {data.documents}" for data in FUNCTION_LIST])

    system = f"""Please construct a systematic plan to generate an optimal response to the user's instruction, utilizing a set of provided tools. Each plan will correspond to an "observation" value which will describe the output of one of the available "functions" given an "input" string that could be the user's question, one or more prior "observation" values, or a combination of both.
    Here are the tools available to be called:
    {functions}

    The output should be a list of plans with "function", "inputs", and "observation" fields in this format:
    
    Plan: [first action to take based on input question]
    :observation0: = ["function" to call with input parameters]
    Plan: [next action to take, based on results of :observation0:]
    :observation1: = [next function to call and input parameters, which may include reference to previous observation, e.g. "Given context :observation0:"]
    """
    # prompt = prompt.replace("{functions}",functions).replace("{query}",query)
    prompt_template = char.compile_text(char.model_data["config"]["prompt_template"].format(
            system=system,
            history="",
            prompt=query
        ))

    generator = char.LLM(
        prompt_template,
        stop=char.model_data["config"]["stop_words"].split(",")+["\n\n\n"],
        # grammar=load_json_grammar(),
        stream=True,
        max_tokens=1024,
        mirostat_mode = 2,
        mirostat_tau = 4.0,
        mirostat_eta = 0.2,
    )
    for response in generator: pass
    args = json.loads(response["choices"][0]["text"])
    print(args)
    return args
    if len(results): # found function
        metadata = results[0]["metadata"]
        return metadata
    else:
        print("failed to find function")
        return None

def get_args(char: "Character", instructions: str, prompt: str, use_grammar=False):
    grammar = load_json_grammar() if use_grammar else ""
    
    try:
        system = "\n\n".join([instructions,char.summarize_context()])
        prompt_template = char.compile_text(char.model_data["config"]["prompt_template"].format(
            system=system,
            history="",
            prompt=prompt
        ))
        
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

def get_plan(char: "Character", instructions: str, prompt: str, use_grammar=False):
    grammar = load_json_grammar() if use_grammar else ""
    
    try:
        system = "\n\n".join([instructions,char.summarize_context()])
        prompt_template = char.compile_text(char.model_data["config"]["prompt_template"].format(
            system=system,
            history="",
            prompt=prompt
        ))
        
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

def call_function(character, prompt: str, reply: str, threshold=1., retries=3, use_grammar=False, verbose=False):
    try:
        # metadata = get_function(character, prompt, threshold, verbose)
        # if metadata is None: metadata = get_function(character, reply, threshold, verbose)
        metadata = get_function(character, reply, threshold, verbose)
        if metadata and metadata["function"] in FUNCTION_MAP:
            while retries>0:
                instructions = metadata['instructions'].format(template=metadata['template'],context=reply)
                args = get_args(character, instructions=instructions, prompt=prompt, use_grammar=use_grammar)
                
                if args:
                    try:
                        is_stopped = args.get("STOP",False)
                        if is_stopped and int(is_stopped):
                            print("character refused to execute function")
                            return None
                        else:
                            args = {k:args[k] for k in args if k in metadata['arguments']}
                            return metadata['function'], args, FUNCTION_MAP[metadata["function"]](**args)
                    except Exception as e:
                        print(e)
                        
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