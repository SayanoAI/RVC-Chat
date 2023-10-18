> :warning: **The main branch is bleeding edge**: Expect frequent updates and many breaking changes after every commit

# RVC Chat
This project allows you to create character cards (like NAI/character.ai/tavern) and model cards to easily chat with a LLM voiced by your favourite RVC model. Use the app to download the required files before using or manually download them here: https://huggingface.co/datasets/SayanoAI/RVC-Studio/tree/main

## Features
* Character card builder: model agnostic JSON character cards compatible with any LLM model
* Model card builder: customize prompt format for each LLM to minimize parroting (e.g. LLM speaking for the user)
* LLM E/RP: uncensored chat with your RVC model in real time using popular GGUF LLMs.

## Planned Features
* multimodal integration: send and receive pictures from your LLM model

## Requirements
- Python 3.8 or higher (developed and tested on v3.8.17)
- [Git](https://git-scm.com/download/win)

## Easy Install
1. Clone this repository or download the zip file and extract it.
2. Double-click "conda-installer.bat" to install the latest version of [conda package manager](https://docs.conda.io/projects/miniconda/en/latest/)
3. Double-click "conda-start.bat" (if you skipped step 2.)

## Manual Installation
1. Clone this repository or download the zip file.
2. Navigate to the project directory and create a virtual environment with the command `virtualenv venv`.
3. Activate the virtual environment with the command `source venv/bin/activate` on Linux/Mac or `venv\Scripts\activate` on Windows. Or use `conda create -n RVC-Chat & conda activate RVC-Chat` if you're using conda package manager.
4. Install the required packages with the command `pip install -r requirements.txt`.
5. Run the streamlit app with the command `streamlit run Home.py`.

## Instructions for Chat page
1. Download one of the following recommended LLMs (or use the homepage downloader):
* [mistral-7b-openorca](https://huggingface.co/TheBloke/Mistral-7B-OpenOrca-GGUF/resolve/main/mistral-7b-openorca.Q4_K_M.gguf)
* [mythalion-13b](https://huggingface.co/TheBloke/Mythalion-13B-GGUF/resolve/main/mythalion-13b.Q4_K_M.gguf)
2. Write your name (this is what the LLM will call you)
3. Select Your Character (or create one using Character Builder)
4. Select a language model, you will have to set up the configuration yourself in the Model Config page if you use your own models
5. Click "Start Chatting" to chat with your model

**Feel free to explore other LLMs. (you will have to build your own model config)**

## Dockerize
Run `docker compose up --build` in the main project folder.

## FAQs
* nothing yet :D

## Disclaimer
This project is for educational and research purposes only. The generated voice overs are not intended to infringe on any copyrights or trademarks of the original songs or text. The project does not endorse or promote any illegal or unethical use of the generative AI technology. The project is not responsible for any damages or liabilities arising from the use or misuse of the generated voice overs.

## Credits
This project uses code and AI models from the following repositories:
- [RVC-Studio](https://github.com/SayanoAI/RVC-Studio) by SayanoAI.
- [SpeechT5: A Self-Supervised Pre-training Model for Speech Recognition and Generation](https://github.com/microsoft/SpeechT5) by Microsoft.

We thank all the authors and contributors of these repositories for their amazing work and for making their code and models publicly available.