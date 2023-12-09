

import random
import subprocess
from time import sleep
from typing import Any, Iterator, List, Optional, Union
import requests
import asyncio
from webui import SERVERS, get_cwd
from webui.utils import pid_is_active, stop_server

CWD = get_cwd()

def start_server(model, host="localhost", port=8000, n_gpu_layers=0, n_ctx=2048):
    pid = SERVERS.LLM_PID
    if pid_is_active(pid):
        if model!=SERVERS.LLM_MODEL: # change model
            stop_server(pid)
            print(f"Switching models: process {pid} killed.")
        else:
            return SERVERS.LLM_URL
    
    base_url = f"http://{host}:{port}/api"
    cmd = f"koboldcpp.exe --model={model} --host={host} --port={port} --gpulayers={n_gpu_layers} --contextsize={n_ctx} --skiplauncher --multiuser --usecublas --quiet"
    print(f"{cmd=}")
    process = subprocess.Popen(cmd, cwd=CWD)
    for i in range(30): # wait for server to start up
        try:
            with requests.get(base_url) as req:
                if req.status_code==200:
                    SERVERS.LLM_URL = base_url
                    SERVERS.LLM_PID = process.pid
                    SERVERS.LLM_MODEL = model
                    break
        except Exception:
            sleep(1.)
            print(f"waited {i+1} seconds...")
    return base_url

class Llama:
    def __init__(self, fname, n_gpu_layers=0, n_ctx=2048, verbose=False):
        self.base_url = start_server(fname, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx)
        self.verbose = verbose

    @property
    def generate_url(self):
        return f"{self.base_url}/v1/generate"
    
    @property
    def check_url(self):
        return f"{self.base_url}/extra/generate/check"
    
    @property
    def tokens_url(self):
        return f"{self.base_url}/extra/tokencount"
    
    @property
    def headers(self):
        return {"Accept": "application/json", "Content-Type": "application/json"}

    def __call__(self, *args: Any, **kwds: Any):
        return self.create_completion(*args,**kwds)

    def token_count(self, prompt):
        try:
            with requests.post(self.tokens_url,json={"prompt": prompt},headers=self.headers) as req:
                result = req.json()
                return result["value"]
        except Exception as e:
            print(e)
            return 0

    def create_completion(
        self,
        prompt: str,
        suffix: Optional[str] = None,
        max_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        logprobs: Optional[int] = None,
        echo: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        top_k: int = 100,
        stream: bool = False,
        tfs_z: float = 1.0,
        mirostat_mode: int = 2,
        mirostat_tau: float = 4.0,
        mirostat_eta: float = 0.2,
        grammar: Optional[str] = "",
        model: str = None,
        **kwargs #extra args
    ):
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate. If max_tokens <= 0, the maximum number of tokens to generate is unlimited and depends on n_ctx.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.

        Raises:
            ValueError: If the requested tokens exceed the context window.
            RuntimeError: If the prompt fails to tokenize or the model fails to evaluate the prompt.

        Returns:
            Response object containing the generated text.
        """
        completion_or_chunks = self._generate(
            prompt=prompt,
            # suffix=suffix,
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            # logprobs=logprobs,
            # echo=echo,
            stop_sequence=stop,
            # frequency_penalty=frequency_penalty,
            # presence_penalty=presence_penalty,
            rep_pen=repeat_penalty,
            rep_pen_range=320,
            rep_pen_slope=0.7,
            top_k=top_k,
            stream=stream,
            tfs=tfs_z,
            min_p=0.05,
            mirostat=mirostat_mode,
            mirostat_tau=mirostat_tau,
            mirostat_eta=mirostat_eta,
            # model=model,
            grammar=grammar,
            top_a=0,
            typical=1,
            n=1,
            **kwargs
        )
        if stream:
            chunks: Iterator = completion_or_chunks
            return chunks
        completion = next(completion_or_chunks)  # type: ignore
        return completion
    
    def _generate(self,stream=False,**kwargs):
        genkey = "KCPP{:08x}".format(random.getrandbits(64))
        data = kwargs
        data["genkey"] = genkey

        async def generate_completion():
            # use run_in_executor to run a blocking function in another thread
            loop = asyncio.get_event_loop()
            # headers={"Content-Type":"application/json"}
            def generate():
                return requests.post(self.generate_url, json=data, headers=self.headers)
            
            result = loop.run_in_executor(None, generate)
            while stream and not result.done():
                with requests.post(self.check_url, json={"genkey": genkey}, headers=self.headers) as response:
                    asyncio.sleep(1.)
                    res = response.json()
                    yield {"choices": res["results"]}
            
            # response = await result
            # res = response.json()
            # yield {"choices": res["results"]}
        
        loop = asyncio.new_event_loop()
        # create an async iterator from the async generator
        iterator = generate_completion().__aiter__()
        while True:
            try:
                # get the next value
                value = loop.run_until_complete(iterator.__anext__())
                # print the value
                yield value
            except StopAsyncIteration:
                # break the loop when the iterator is exhausted
                break
        # close the event loop
        loop.close()