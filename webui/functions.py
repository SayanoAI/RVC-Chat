from functools import lru_cache, wraps
import io
import json
import os
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from webui.chat import Character
    from webui.vector_db import VectorDB
from webui.downloader import BASE_MODELS_DIR
from webui.image_generation import NSFW_KEYWORDS, generate_prompt
from webui import ObjectNamespace

FUNCTION_MAP = ObjectNamespace(
    create_image = dict(
        call = generate_prompt,
        description = "tool to enable {{name}} to draw a new image based on {{user}}'s request",
        arguments = ["subject","description","emotion","environment","negative","orientation","seed","steps","cfg","scale","style"],
        documents = """Generate a picture of a blue sky with white clouds and a rainbow.
Create an image of a cat wearing a hat and sunglasses, cartoon style.
Draw an image of a forest with a river and a deer, watercolor style.
Make an image of a pizza with cheese, pepperoni, and mushrooms, realistic style.
Produce an image of a city skyline at night with lights and stars, digital art style.
Render an image of a flower garden with roses, tulips, and sunflowers, oil painting style.
Synthesize an image of a spaceship flying in space with planets and asteroids, sci-fi style.
Compose an image of a beach with palm trees, sand, and waves, sketch style.
Design an image of a dragon breathing fire, fantasy style.
Craft an image of a snowman with a scarf, hat, and carrot nose, pixel art style.
can you draw me a [object]?
Show me what [noun/pronoun/action] looks like.
Send me a picture of [object].
Draw an [object] for me.
SUBJECT: [words to describe the main image subject]\nDESCRIPTION: [words to describe the main subject's appearance]\nENVIRONMENT: [words to describe the environment or items in it]\nNEGATIVE: [words to describe things to remove from the image]""",
        metadata = dict(
            subject = "string containing main subject of the image (e.g. a playful cat, a beautiful woman, an old man, etc.)",
            description = "string describing the subject's physical appearance (e.g. hair color, eye color, clothes, etc.)",
            emotion = "string describing the overall emotion of the subject or environment (e.g. sombre, smile, happy, etc.)",
            environment = "string describing everything else you want to include in the image (e.g. animals, environment, background, etc.)",
            negative = "string containing anything the user wants to fix or remove from the drawing (e.g. extra fingers, missing limbs, errors, etc.)",
            orientation = "string describing the orientation of the image [square,portrait,landscape]",
            seed = "number to initialize the image with (use -1 for random)",
            steps = "number of steps to sample (20-40)",
            cfg = "number to show how closely to follow the prompt (7.0-12.0)",
            scale="number to scale the image by (1.0-2.0)",
            style="string to describe the style of image (e.g. drawing, photograph, sketch, etc.)",
        ),
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    modify_image = dict(
        call = generate_prompt,
        description = "tool to enable {{name}} to change an existing image based on {{user}}'s request",
        arguments = ["image","subject","description","emotion","environment","negative","steps","cfg","change_ratio"],
        documents = """Alter the picture of a dog to make it look like a cat, realistic style.
Change the picture of a landscape to add a waterfall and a rainbow, watercolor style.
Transform the picture of a person to change their hair color and style, cartoon style.
Convert the picture of a car to make it look like a spaceship, sci-fi style.
Recast the picture of a flower to make it look like a butterfly, fantasy style.
Revise the picture of a building to add graffiti and posters, digital art style.
Vary the picture of a cake to make it look like a pizza, realistic style.
Adapt the picture of a map to add labels and icons, sketch style.
Customize the picture of a logo to make it look like a different brand, pixel art style.
Reorganize the picture of a bird to make it look like a dragon, fantasy style.
please redraw your image [with/without these conditions].
Can you fix this [conditions]?
There's [something wrong with the image]...
Please change [the image] to [something else].
Can you make [things to change in the image]?
Can you draw [object with conditions]""",
        metadata = dict(
            subject = "string containing main subject of the image (e.g. a playful cat, a beautiful woman, an old man, etc.)",
            description = "string describing the subject's physical appearance (e.g. hair color, eye color, clothes, etc.)",
            emotion = "string describing the overall emotion of the subject or environment (e.g. sombre, smile, happy, etc.)",
            environment = "string describing everything else you want to include in the image (e.g. animals, environment, background, etc.)",
            negative = "string containing anything the user wants to fix or remove from the drawing (e.g. extra fingers, missing limbs, errors, etc.)",
            steps = "number of steps to sample (20-40)",
            cfg = "number to show how closely to follow the prompt (7.0-12.0)",
            change_ratio="string to describe how much to alter the image (small|medium|large)",
        ),
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    resize_image = dict(
        call = generate_prompt,
        description = "tool to enable {{name}} to resize an image based on {{user}}'s request",
        documents = """Can you please resize this image?
        Make the above image larger.
        Scale it up.""",
        arguments = ["image","scale","steps","cfg"],
        metadata = dict(
            scale="number to scale the image by (1.0-2.0)",
            steps = "number of steps to sample (20-40)",
            cfg = "number to show how closely to follow the prompt (7.0-12.0)",
        ),
        instructions="Construct a JSON object with the following fields: {template}. Use the content below to build the JSON object.\n\n{context}"
    ),
    # respond_to_user = dict(
    #     call = None,
    #     description = "tool to enable {{name}} to respond to {{user}} while staying in character",
    # ),
)

TOOLKIT_LIST = "\n".join(f"[*] {name}: {metadata.description}" for name,metadata in FUNCTION_MAP.items())

def retry(times, exceptions=Exception): # define the decorator with parameters
    def decorator (func): # take the function to be wrapped as an argument
        @wraps (func) # use wraps to copy the original function's name, docstring, etc.
        def wrapper (*args, **kwargs): # define the wrapper function
            attempt = 0 # initialize the attempt counter
            while attempt < times: # loop until the maximum number of retries is reached
                try: # try to execute the original function with the given arguments
                    return func (*args, **kwargs) # return the result if successful
                except exceptions as e: # catch the specified exceptions
                    print (f"Attempt {attempt + 1} of {times} failed due to {e}") # print the error message
                finally: attempt += 1 # increment the attempt counter
            return func(*args, **kwargs) # try one last time and return the result or raise the exception
        return wrapper # return the wrapper function
    return decorator # return the decorator

@retry(times=3)
def get_function(char: "Character", query: str, threshold=1.):
    grammar = load_json_grammar()

    results = char.ltm.get_query(
        query=query,
        include=["metadatas", "distances"],
        type="function",
        threshold=threshold,
        n_results=len(FUNCTION_MAP),
        verbose=True)

    if len(results):
        tools = "\n".join(f"[*] {result['metadata']['function']}: {result['metadata']['description']}" for result in results)
        options = {"tools": ["name of first tool","name of second tool",...],"observations": ["result from first tool","result from second tool"]}
        
        system = "Choose at least one of the following tools based on the interactions between {{user}} and {{name}}:\n\n{{tools}}"
        history = char.compile_chat_history(
            char.messages[char.context_index:] + [
                dict(role="SYSTEM",
                content=f"Please structure your response as a JSON object using the following format: {options}.")
            ])
        prompt_template = char.model_data["config"]["prompt_template"].format(
            system=system,
            history=history,
            prompt=query
        )
        
        generator = char.LLM(
            char.compile_text(prompt_template,tools=tools),
            stop=char.model_data["config"]["stop_words"].split(",")+["\n\n\n"],
            grammar=grammar,
            stream=True,
            max_tokens=128,
            mirostat_mode = 2,
            mirostat_tau = 4.0,
            mirostat_eta = 0.2,
        )
        for response in generator: pass
        selected_tools = json.loads(response["choices"][0]["text"])["tools"]
        
        if len(selected_tools) and selected_tools[0] in FUNCTION_MAP:  # do multi tool later
            tool = selected_tools.pop()
            print(f"get_function: {tool=}")
            return FUNCTION_MAP[tool]
    
    return None
    
@retry(times=3)
def get_args(char: "Character", system: str, history: str, prompt: str, use_grammar=False):
    grammar = load_json_grammar() if use_grammar else ""
    
    prompt_template = char.model_data["config"]["prompt_template"].format(
        system=system,
        history=history,
        prompt=prompt
    )
    
    generator = char.LLM(
        char.compile_text(prompt_template),
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
    print(f"get_args: {args=}")
    return args

@lru_cache
def load_json_grammar(fname=os.path.join(BASE_MODELS_DIR,"LLM","json.gbnf")):
    with open(fname,"r") as f:
        grammar = f.read()

    print(f"{grammar=}")
    return grammar

def call_function(character: "Character", prompt: str, threshold=1., retries=3, use_grammar=False, **kwargs):
    try:
        errors = []
        function = get_function(character, prompt, threshold)
        if function and function.call:
            system = function.instructions.format(template=vars(function.metadata),context=character.context)
            history = character.compile_chat_history([dict(role=character.user,content=prompt)])
            while retries>0:
                try:
                    error_prompt = f"Please avoid the following errors when generating your response: {', '.join(errors)}" if len(errors) else ""
                    censored_prompt = f"Do not use the following words when generating your response: {', '.join(NSFW_KEYWORDS)}" if kwargs.get("censor") else ""
                    args = get_args(character,
                                    system=system,
                                    history=history,
                                    prompt="\n".join(i for i in [prompt,censored_prompt,error_prompt] if i),
                                    use_grammar=use_grammar)
                    
                    if args and len(args):
                        args = {k:args[k] for k in args if k in function.arguments}
                        print(f"call_function: {args=}")

                        args.update(kwargs)

                        if "image" in function.arguments:
                            image = kwargs.pop("image")
                            image = character.get_image if image is None else image
                            if image:
                                args["image"] = image
                                image = Image.open(io.BytesIO(image))
                                args["width"] = image.width
                                args["height"] = image.height
                                
                        results = function.call(**args) if function.call else None
                        
                        if results is not None: return results
                except Exception as e:
                    print(e)
                    errors.append(e)
                finally: retries-=1
    except Exception as e:
        print(e)

    return None

def load_functions(vdb: "VectorDB"):
    for name, metadata in FUNCTION_MAP.items():
        vdb.add_function(
            documents=metadata.documents,
            function=name,
            arguments=metadata.arguments,
            description=metadata.description
            )