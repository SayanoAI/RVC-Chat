@echo off

REM This script assumes that you have git installed and added to your PATH
REM It also assumes that you have a .cache folder in your current directory
REM If not, you may need to create it or change the destination folder
REM You may also need to change the branch or tag of the ComfyUI repo you want to install

git clone https://github.com/comfyanonymous/ComfyUI .cache/ComfyUI
REM This command clones the main branch of the ComfyUI repo into the .cache/ComfyUI folder
REM You can check the status of the clone with git status
REM You can also run other git commands to update or modify the repo as needed pause REM This command pauses the script and waits for a key press to continue exit REM This command exits the script