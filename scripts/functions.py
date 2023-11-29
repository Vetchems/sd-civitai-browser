import requests
import json
import time
import threading
import urllib.request
import urllib.error
import os
from tqdm import tqdm
import re
from requests.exceptions import ConnectionError
import shutil
import tempfile
import gradio as gr

# Set the URL for the API endpoint
api_url = "https://civitai.com/api/v1/models?limit=50"
json_data = None

def create_dummy(file_name):
    dummy_path = get_dummy_path(file_name)
    os.makedirs(os.path.dirname(dummy_path), exist_ok=True)
    with open(dummy_path, 'w') as f:
        f.write('dummy')
        
def get_dummy_path(file_name):
    """Get the path to the dummy file for the given file_name"""
    dir_name = os.path.dirname(file_name)
    dir_name = os.path.abspath(dir_name)
    return os.path.join(dir_name,file_name + ".dummy")
        
def remove_dummy(file_name):
    dummy_path = get_dummy_path(file_name)
    if os.path.exists(dummy_path):
        os.remove(dummy_path)
        
def check_dummy(file_name):
    return os.path.exists(get_dummy_path(file_name))

def remove_empty_directories(path):
    if not os.path.isdir(path):
        return
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                remove_empty_directories(fullpath)
    files = os.listdir(path)
    if len(files) == 0:
        print("Removing empty directory:", path)
        os.rmdir(path)
        
def download_file(url, file_name):
    # Maximum number of retries
    max_retries = 5

    # Delay between retries (in seconds)
    retry_delay = 10
    if os.path.exists(file_name):
        # skip if exists
        return
    
    if check_dummy(file_name):
        return
    
    create_dummy(file_name)
    try:
        dest = file_name
        file_name = tempfile.NamedTemporaryFile().name

        while True:
            # Check if the file has already been partially downloaded
            if os.path.exists(file_name):
                # Get the size of the downloaded file
                downloaded_size = os.path.getsize(file_name)

                # Set the range of the request to start from the current size of the downloaded file
                headers = {"Range": f"bytes={downloaded_size}-"}
            else:
                downloaded_size = 0
                headers = {}

            # Split filename from included path
            tokens = re.split(re.escape('\\'), dest)
            file_name_display = tokens[-1]

            # Initialize the progress bar
            progress = tqdm(total=1000000000, unit="B", unit_scale=True, desc=f"Downloading {file_name_display}", initial=downloaded_size, leave=False)

            # Open a local file to save the download
            with open(file_name, "ab") as f:
                while True:
                    try:
                        # Send a GET request to the URL and save the response to the local file
                        response = requests.get(url, headers=headers, stream=True)

                        # Get the total size of the file
                        total_size = int(response.headers.get("Content-Length", 0))

                        # Update the total size of the progress bar if the `Content-Length` header is present
                        if total_size == 0:
                            total_size = downloaded_size
                        progress.total = total_size 

                        # Write the response to the local file and update the progress bar
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:  # filter out keep-alive new chunks
                                f.write(chunk)
                                progress.update(len(chunk))

                        downloaded_size = os.path.getsize(file_name)
                        # Break out of the loop if the download is successful
                        break
                    except ConnectionError as e:
                        # Decrement the number of retries
                        max_retries -= 1

                        # If there are no more retries, raise the exception
                        if max_retries == 0:
                            # remove temp file
                            if os.path.exists(file_name):
                                os.remove(file_name)
                            remove_dummy(dest)
                            raise ConnectionError(f"Failed to download from {url}.") from e

                        # Wait for the specified delay before retrying
                        time.sleep(retry_delay)

            # Close the progress bar
            progress.close()
            downloaded_size = os.path.getsize(file_name)
            # Check if the download was successful
            if downloaded_size >= total_size:
                print(f"{file_name_display} successfully downloaded.")
                # move to dest
                shutil.move(file_name, dest)
                break
            else:
                print(f"Error: File download failed. Retrying... {file_name_display}")
        if os.path.exists(file_name):
            os.remove(file_name)
    finally:
        remove_dummy(dest)
        # clean up empty directories
        remove_empty_directories(os.path.dirname(dest))

def replace_invalid_chars(file_name):
    first_processed = file_name.replace(" ","_").replace("(","").replace(")","").replace("|","").replace(":","-")
    # remove invalid chars for windows
    return ''.join(c for c in first_processed if c not in r'<>:"/\|?*')

def wrapped_download_file_thread(url, file_name:str, content_type, use_new_folder, model_name):
    """
    Wrapped function that process args.
    model name can be empty, in which case it is extracted from file_name
    """
    # check if it is valid url
    if not url or not url.startswith("http"):
        return f"No valid URL provided: {url}"
    # check if filename has extensions, if not, add .safetensors
    if not file_name:
        return f"No file name provided: {file_name}"
    file_name = replace_invalid_chars(file_name)
    if "." not in file_name:
        file_name = file_name + ".safetensors"
    if content_type not in ["Checkpoint", "Hypernetwork", "TextualInversion", "AestheticGradient", "VAE", "LORA", "LoCon"]:
        return f"Invalid content type, given {content_type} but expected one of ['Checkpoint', 'Hypernetwork', 'TextualInversion', 'AestheticGradient', 'VAE', 'LORA', 'LoCon']"
    if not model_name:
        # remove ext from file name
        if "." in file_name:
            model_name = file_name[:file_name.rindex(".")]
        else:
            model_name = file_name
    # replace invalid chars in model name
    model_name = replace_invalid_chars(model_name).replace(".", "_")
    download_file_thread(url, file_name, content_type, use_new_folder, model_name)
    return f"Downloading {model_name}..."

def download_file_thread(url, file_name, content_type, use_new_folder, model_name):
    """
    Download the file and save it to a local file
    
    @param url:string The URL of the file to download, example) https://example.com/file.txt (may or may not contain file name)
    @param file_name:string The name of the file to save the download to example) file.txt
    @param content_type:string The type of content being downloaded, example) Checkpoint, Hypernetwork, TextualInversion, AestheticGradient, VAE, LORA, LoCon
    @param use_new_folder:boolean Whether to save the file to a new folder or not (default: False)
    @param model_name:string The name of the model being downloaded, used for subfolder (default: None or use file_name)

    """
    model_name = replace_invalid_chars(model_name)
    if content_type == "Checkpoint":
        folder = "models/Stable-diffusion"
        new_folder = "models/Stable-diffusion/new"
    elif content_type == "Hypernetwork":
        folder = "models/hypernetworks"
        new_folder = "models/hypernetworks/new"
    elif content_type == "TextualInversion":
        folder = "embeddings"
        new_folder = "embeddings/new"
    elif content_type == "AestheticGradient":
        folder = "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings"
        new_folder = "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings/new"
    elif content_type == "VAE":
        folder = "models/VAE"
        new_folder = "models/VAE/new"
    elif content_type == "LORA" or content_type == "LoCon":
        folder = "models/Lora"
        new_folder = "models/Lora/new"
    if content_type == "TextualInversion" or content_type == "VAE" or content_type == "AestheticGradient":
        if use_new_folder:
            model_folder = new_folder
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            
        else:
            model_folder = folder
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
    else:            
        if use_new_folder:
            model_folder = os.path.join(new_folder,model_name)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            
        else:
            model_folder = os.path.join(folder,model_name)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

    path_to_new_file = os.path.join(model_folder, file_name)     

    thread = threading.Thread(target=download_file, args=(url, path_to_new_file))

        # Start the thread
    thread.start()
    return thread # return the thread so we can wait for it to finish if we want

def save_text_file(file_name, content_type, use_new_folder, trained_words, model_name):
    model_name = replace_invalid_chars(model_name)
    print("Save Text File Clicked")
    if content_type == "Checkpoint":
        folder = "models/Stable-diffusion"
        new_folder = "models/Stable-diffusion/new"
    elif content_type == "Hypernetwork":
        folder = "models/hypernetworks"
        new_folder = "models/hypernetworks/new"
    elif content_type == "TextualInversion":
        folder = "embeddings"
        new_folder = "embeddings/new"
    elif content_type == "AestheticGradient":
        folder = "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings"
        new_folder = "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings/new"
    elif content_type == "VAE":
        folder = "models/VAE"
        new_folder = "models/VAE/new"
    elif content_type == "LORA":
        folder = "models/Lora"
        new_folder = "models/Lora/new"
    elif content_type == "LoCon":
        folder = "models/Lora"
        new_folder = "models/Lora/new"
    if content_type == "TextualInversion" or content_type == "VAE" or content_type == "AestheticGradient":
        if use_new_folder:
            model_folder = new_folder
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            
        else:
            model_folder = folder
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
    else:            
        if use_new_folder:
            model_folder = os.path.join(new_folder,model_name)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            
        else:
            model_folder = os.path.join(folder,model_name)
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
   
    path_to_new_file = os.path.join(model_folder, file_name.replace(".ckpt",".txt").replace(".safetensors",".txt").replace(".pt",".txt").replace(".yaml",".txt"))
    if not os.path.exists(path_to_new_file):
        with open(path_to_new_file, 'w') as f:
            f.write(trained_words)

def api_to_data(content_type, sort_type, use_search_term, search_term=None):
    if use_search_term and search_term:
        search_term = search_term.replace(" ","%20")
        return request_civit_api(f"{api_url}&types={content_type}&sort={sort_type}&query={search_term}")
    else:
        return request_civit_api(f"{api_url}&types={content_type}&sort={sort_type}")

def api_next_page(next_page_url=None):
    global json_data
    try: json_data['metadata']['nextPage']
    except: return
    if json_data['metadata']['nextPage'] is not None:
        next_page_url = json_data['metadata']['nextPage']
    if next_page_url is not None:
        return request_civit_api(next_page_url)

def update_next_page(show_nsfw):
    global json_data
    json_data = api_next_page()
    model_dict = {}
    try: json_data['items']
    except TypeError: return gr.Dropdown.update(choices=[], value=None)
    if show_nsfw:
        for item in json_data['items']:
            model_dict[item['name']] = item['name']
    else:
        for item in json_data['items']:
            temp_nsfw = item['nsfw']
            if not temp_nsfw:
                model_dict[item['name']] = item['name']
    return gr.Dropdown.update(choices=[v for k, v in model_dict.items()], value=None), gr.Dropdown.update(choices=[], value=None)

def update_model_list(content_type, sort_type, use_search_term, search_term, show_nsfw):
    global json_data
    json_data = api_to_data(content_type, sort_type, use_search_term, search_term)
    model_dict = {}
    if show_nsfw:
        for item in json_data['items']:
            model_dict[item['name']] = item['name']
    else:
        for item in json_data['items']:
            temp_nsfw = item['nsfw']
            if not temp_nsfw:
                model_dict[item['name']] = item['name']

    return gr.Dropdown.update(choices=[v for k, v in model_dict.items()], value=None), gr.Dropdown.update(choices=[], value=None)

def update_model_versions(model_name=None):
    if model_name is not None:
        global json_data
        versions_dict = {}
        for item in json_data['items']:
            if item['name'] == model_name:

                for model in item['modelVersions']:
                    versions_dict[model['name']] = item["name"]
        return gr.Dropdown.update(choices=[k + ' - ' + v for k, v in versions_dict.items()], value=f'{next(iter(versions_dict.keys()))} - {model_name}')
    else:
        return gr.Dropdown.update(choices=[], value=None)

def update_dl_url(model_name=None, model_version=None, model_filename=None):
    if model_filename:
        global json_data
        dl_dict = {}
        dl_url = None
        model_version = model_version.replace(f' - {model_name}','').strip()
        for item in json_data['items']:
            if item['name'] == model_name:
                for model in item['modelVersions']:
                    if model['name'] == model_version:
                        for file in model['files']:
                            if file['name'] == model_filename:
                                dl_url = file['downloadUrl']
        return gr.Textbox.update(value=dl_url)
    else:
        return gr.Textbox.update(value=None)

def update_model_info(model_name=None, model_version=None):
    if model_name and model_version:
        model_version = model_version.replace(f' - {model_name}','').strip()
        global json_data
        output_html = ""
        output_training = ""
        img_html = ""
        model_desc = ""
        dl_dict = {}
        for item in json_data['items']:
            if item['name'] == model_name:
                model_uploader = item['creator']['username']
                if item['description']:
                    model_desc = item['description']
                for model in item['modelVersions']:
                    if model['name'] == model_version:
                        if model['trainedWords']:
                            output_training = ", ".join(model['trainedWords'])

                        for file in model['files']:
                            dl_dict[file['name']] = file['downloadUrl']

                        model_url = model['downloadUrl']
                        #model_filename = model['files']['name']

                        img_html = '<HEAD><style>img { display: inline-block; }</style></HEAD><div class="column">'
                        for pic in model['images']:
                            img_html = img_html + f'<img src={pic["url"]} width=400px></img>'
                        img_html = img_html + '</div>'
                        output_html = f"<p><b>Model:</b> {model_name}<br><b>Version:</b> {model_version}<br><b>Uploaded by:</b> {model_uploader}<br><br><a href={model_url}><b>Download Here</b></a></p><br><br>{model_desc}<br><div align=center>{img_html}</div>"

        return gr.HTML.update(value=output_html), gr.Textbox.update(value=output_training), gr.Dropdown.update(choices=[k for k, v in dl_dict.items()], value=next(iter(dl_dict.keys())))
    else:
        return gr.HTML.update(value=None), gr.Textbox.update(value=None), gr.Dropdown.update(choices=[], value=None)

def request_civit_api(api_url=None):
    # Make a GET request to the API
    response = requests.get(api_url)

    # Check the status code of the response
    if response.status_code != 200:
      print("Request failed with status code: {}".format(response.status_code))
      exit()

    data = json.loads(response.text)
    return data

def update_everything(list_models, list_versions, model_filename, dl_url):
    (a, d, f) = update_model_info(list_models, list_versions)
    dl_url = update_dl_url(list_models, list_versions, f['value'])
    return (a, d, f, list_versions, list_models, dl_url)

def save_image_files(preview_image_html, model_filename, list_models, content_type):
    print("Save Images Clicked")
    img_urls = re.findall(r'src=[\'"]?([^\'" >]+)', preview_image_html)
    
    name = os.path.splitext(model_filename)[0]

    if content_type == "Checkpoint":
        folder = "models/Stable-diffusion"
    elif content_type == "Hypernetwork":
        folder = "models/hypernetworks"
    elif content_type == "TextualInversion":
        folder = "embeddings"
    elif content_type == "AestheticGradient":
        folder = "extensions/stable-diffusion-webui-aesthetic-gradients/aesthetic_embeddings"
    elif content_type == "VAE":
        folder = "models/VAE"
    elif content_type == "LORA":
        folder = "models/Lora"
    elif content_type == "LoCon":
        folder = "models/Lora"
    
    model_folder = os.path.join(folder,replace_invalid_chars(list_models))

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    for i, img_url in enumerate(img_urls):
        filename = f'{name}_{i}.png'
        img_url = img_url.replace("https", "http")

        print(img_url, filename)
        # Permanent Redirect fix
        save_success = False
        try:
            with urllib.request.urlopen(img_url) as url:
                with open(os.path.join(model_folder, filename), 'wb') as f:
                    f.write(url.read())
                    print("\t\t\tDownloaded")

                #for the first one, let's make an image name that works with preview
                if i == 0:
                    shutil.copy(os.path.join(model_folder, filename), os.path.join(model_folder, name + ".png") )
                    save_success = True
        except urllib.error.URLError as e:
            print(f'Error: {e.reason}')
        if not save_success:
            # use requests.get to download the image
            response = requests.get(img_url)
            if not response.ok:
                print(f'Error: {response.reason}')
                continue
            image_ext = response.headers['Content-Type'].split('/')[-1]
            filename = f'{name}_{i}.{image_ext}'
            with open(os.path.join(model_folder, filename), 'wb') as f:
                f.write(response.content)
                print("\t\t\tDownloaded")
