import gradio as gr
from modules import script_callbacks
from scripts.functions import *

def on_ui_tabs_called():
    with gr.Blocks(analytics_enabled=False) as civitai_interface:
        with gr.Tabs(elem_id="civitai-tabs"):
            with gr.TabItem("CivitAi-Browser"):
                with gr.Row():
                    with gr.Column(scale=2):
                        content_type = gr.Radio(label='Content type:', choices=["Checkpoint","Hypernetwork","TextualInversion","AestheticGradient", "VAE", "LORA", "LoCon"], value="Checkpoint", type="value")
                    with gr.Column(scale=2):
                        sort_type = gr.Radio(label='Sort List by:', choices=["Newest","Most Downloaded","Highest Rated","Most Liked"], value="Newest", type="value")
                    with gr.Column(scale=1):
                        show_nsfw = gr.Checkbox(label="Show NSFW", value=True)
                with gr.Row():
                    use_search_term = gr.Checkbox(label="Search by term?", value=True)
                    search_term = gr.Textbox(label="Search Term", interactive=True, lines=1)
                with gr.Row():
                    get_list_from_api = gr.Button(label="Get List", value="Get List")
                    get_next_page = gr.Button(value="Next Page")
                with gr.Row():
                    list_models = gr.Dropdown(label="Model", choices=[], interactive=True, elem_id="quicksettings", value=None)
                    list_versions = gr.Dropdown(label="Version", choices=[], interactive=True, elem_id="quicksettings", value=None)
                with gr.Row():
                    txt_list = ""
                    dummy = gr.Textbox(label='Trained Tags (if any)', value=f'{txt_list}', interactive=True, lines=1)
                    model_filename = gr.Dropdown(label="Model Filename", choices=[], interactive=True, value=None)
                    dl_url = gr.Textbox(label="Download Url", interactive=False, value=None)
                with gr.Row():
                    update_info = gr.Button(value='1st - Get Model Info')
                    save_text = gr.Button(value="2nd - Save Text")
                    save_images = gr.Button(value="3rd - Save Images")
                    download_model = gr.Button(value="4th - Download Model")
                    save_model_in_new = gr.Checkbox(label="Save Model to new folder", value=False)
                with gr.Row(elem_id="html_row"):
                    preview_image_html = gr.HTML()
                save_text.click(
                    fn=save_text_file,
                    inputs=[
                    model_filename,
                    content_type,
                    save_model_in_new,
                    dummy,
                    list_models,
                    ],
                    outputs=[]
                )
                save_images.click(
                    fn=save_image_files,
                    inputs=[
                    preview_image_html,
                    model_filename,
                    list_models,
                    content_type
                    ],
                    outputs=[]
                )
                download_model.click(
                    fn=download_file_thread,
                    inputs=[
                    dl_url,
                    model_filename,
                    content_type,
                    save_model_in_new,
                    list_models,
                    ],
                    outputs=[]
                )
                get_list_from_api.click(
                    fn=update_model_list,
                    inputs=[
                    content_type,
                    sort_type,
                    use_search_term,
                    search_term,
                    show_nsfw,
                    ],
                    outputs=[
                    list_models,
                    list_versions,
                    ]
                )
                update_info.click(
                    fn=update_everything,
                    #fn=update_model_info,
                    inputs=[
                    list_models,
                    list_versions,
                    model_filename,
                    dl_url
                    ],
                    outputs=[
                    preview_image_html,
                    dummy,
                    model_filename,
                    list_versions,
                    list_models,
                    dl_url
                    ]
                )
                list_models.change(
                    fn=update_model_versions,
                    inputs=[
                    list_models,
                    ],
                    outputs=[
                    list_versions,
                    ]
                )

                list_versions.change(
                    fn=update_model_info,
                    inputs=[
                    list_models,
                    list_versions,
                    ],
                    outputs=[
                    preview_image_html,
                    dummy,
                    model_filename,
                    ]
                )
                model_filename.change(
                    fn=update_dl_url,
                    inputs=[list_models, list_versions, model_filename,],
                    outputs=[dl_url,]
                )
                get_next_page.click(
                    fn=update_next_page,
                    inputs=[
                    show_nsfw,
                    ],
                    outputs=[
                    list_models,
                    list_versions,
                    ]
                )
            with gr.TabItem("Manual-CivitAi-Download"):
                with gr.Row():
                    # Manual Tab, you can input your own URL and file name.
                    input_url_textbox = gr.Textbox(label="URL", interactive=True, lines=1)
                    input_filename_textbox = gr.Textbox(label="File Name", interactive=True, lines=1)
                    content_type_dropdown = gr.Dropdown(label="Content Type",
                                                        choices=["Checkpoint","Hypernetwork","TextualInversion","AestheticGradient", "VAE", "LORA", "LoCon"],
                                                        interactive=True, value="Checkpoint")
                    use_new_folder_checkbox = gr.Checkbox(label="Save to new folder", value=False)
                    input_foldername_textbox = gr.Textbox(label="Folder Name(Optional)", interactive=True, lines=1) # can be empty
                    download_button = gr.Button(value="Download")

                    debug_result_textbox = gr.Textbox(label="Debug Result", interactive=False, lines=1)
                    download_button.click(
                        fn=wrapped_download_file_thread,
                        inputs=[
                            input_url_textbox,
                            input_filename_textbox,
                            content_type_dropdown,
                            use_new_folder_checkbox,
                            input_foldername_textbox,
                        ],
                        outputs=[
                            debug_result_textbox,
                        ]
                    )
    return (civitai_interface, "CivitAi", "civitai-interface"),

script_callbacks.on_ui_tabs(on_ui_tabs_called)