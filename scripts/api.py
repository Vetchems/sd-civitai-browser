"""
Registers API routes for the webui
"""
import gradio as gr
from fastapi import FastAPI, Form
from pydantic import BaseModel
from typing import List, Optional, Tuple, Union
import threading
from scripts.civitaiapi import download_file_thread


### ====================classes========================
class DownloadRequestResponse(BaseModel):
    """
    Basic model download response
    """
    message:str
    success:bool

### ====================functions======================
def assert_download_conditions(url:str, file_name:str, content_type:str, use_new_folder:bool, model_name:Optional[str]=None) -> Union[DownloadRequestResponse, Tuple]:
    """
    Asserts conditions for download. Returns DownloadRequestResponse if conditions not met, else returns refined args
    """
    if not url:
        return DownloadRequestResponse(message="No URL provided", success=False)
    if not file_name:
        return DownloadRequestResponse(message="No file name provided", success=False)
    # check content_type in Checkpoint, Hypernetwork, TextualInversion, AestheticGradient, VAE, LORA, LoCon
    if content_type not in ["Checkpoint", "Hypernetwork", "TextualInversion", "AestheticGradient", "VAE", "LORA", "LoCon"]:
        return DownloadRequestResponse(
            message=f"Invalid content type, given {content_type} but expected one of ['Checkpoint', 'Hypernetwork', 'TextualInversion', 'AestheticGradient', 'VAE', 'LORA', 'LoCon']", success=False)
    if not model_name:
        # remove ext from file name
        if "." in file_name:
            model_name = file_name[:file_name.rindex(".")]
        else:
            model_name = file_name
    return url, file_name, content_type, use_new_folder, model_name

def wrapped_download_file_thread(url:str, model_name:str, file_name:str, content_type:str, use_new_folder:bool=False, wait:bool=False) -> DownloadRequestResponse:
    """
    Wraps download_file_thread to return DownloadRequestResponse
    """
    assert_download_conditions_response = assert_download_conditions(url, file_name, content_type, use_new_folder, model_name)
    if isinstance(assert_download_conditions_response, DownloadRequestResponse):
        return assert_download_conditions_response
    url, file_name, content_type, use_new_folder, model_name = assert_download_conditions_response
    thread:threading.Thread = download_file_thread(url, file_name, content_type, use_new_folder, model_name) # started thread
    if wait:
        thread.join()
        return DownloadRequestResponse(message=f"Downloaded {model_name}", success=True)
    return DownloadRequestResponse(message=f"Downloading {model_name}...", success=True)

def register_download_api(app:FastAPI):
    @app.post("/download/model", response_model=DownloadRequestResponse)
    def download_model(url:str=Form(""), model_name:str=Form(""), file_name:str=Form(""), content_type:str=Form(""), use_new_folder:bool=Form(False), wait:bool=Form(False)):
        """
        Download a model from a URL
        example : curl -X POST "http://localhost:7860/download/model" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "url=https://www.example.com/model.zip" -F "model_name=example_model" -F "file_name=example_model.zip" -F "content_type=Checkpoint" -F "use_new_folder=false" -F "wait=false"
        """
        return wrapped_download_file_thread(url, model_name, file_name, content_type, use_new_folder, wait)

def register_api(_:gr.Blocks, app:FastAPI):
    """
    Registers hooks for app on webui startup
    """
    register_download_api(app)


# only works in context of sdwebui
try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(register_api)
except (ImportError, ModuleNotFoundError) as e:
    print("Could not bind uploader-api to app")
