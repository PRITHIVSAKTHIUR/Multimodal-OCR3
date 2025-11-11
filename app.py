import os
import sys
import random
import uuid
import json
import time
from threading import Thread
from typing import Iterable
from huggingface_hub import snapshot_download

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoProcessor,
    TextIteratorStreamer,
)

from transformers.image_utils import load_image
from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.steel_blue = colors.Color(
    name="steel_blue",
    c50="#EBF3F8",
    c100="#D3E5F0",
    c200="#A8CCE1",
    c300="#7DB3D2",
    c400="#529AC3",
    c500="#4682B4",
    c600="#3E72A0",
    c700="#36638C",
    c800="#2E5378",
    c900="#264364",
    c950="#1E3450",
)

class SteelBlueTheme(Soft):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.steel_blue,
        neutral_hue: colors.Color | str = colors.slate,
        text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_800)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_500)",
            button_secondary_text_color="black",
            button_secondary_text_color_hover="white",
            button_secondary_background_fill="linear-gradient(90deg, *primary_300, *primary_300)",
            button_secondary_background_fill_hover="linear-gradient(90deg, *primary_400, *primary_400)",
            button_secondary_background_fill_dark="linear-gradient(90deg, *primary_500, *primary_600)",
            button_secondary_background_fill_hover_dark="linear-gradient(90deg, *primary_500, *primary_500)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px",
            color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

steel_blue_theme = SteelBlueTheme()

css = """
#main-title h1 {
    font-size: 2.3em !important;
}
#output-title h2 {
    font-size: 2.1em !important;
}
"""

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("torch.version.cuda =", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

print("Using device:", device)

MAX_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Chandra-OCR
MODEL_ID_V = "datalab-to/chandra"
processor_v = AutoProcessor.from_pretrained(MODEL_ID_V, trust_remote_code=True)
model_v = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID_V,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

# Load Nanonets-OCR2-3B
MODEL_ID_X = "nanonets/Nanonets-OCR2-3B"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).to(device).eval()

# Load Dots.OCR from the local, patched directory
MODEL_PATH_D = "prithivMLmods/Dots.OCR-Latest-BF16" # -> alt of [rednote-hilab/dots.ocr]
processor_d = AutoProcessor.from_pretrained(MODEL_PATH_D, trust_remote_code=True)
model_d = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH_D,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
).eval()

# Load olmOCR-2-7B-1025
MODEL_ID_M = "allenai/olmOCR-2-7B-1025"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to(device).eval()

@spaces.GPU
def generate_image(model_name: str, text: str, image: Image.Image,
                   max_new_tokens: int, temperature: float, top_p: float,
                   top_k: int, repetition_penalty: float):
    """
    Generates responses using the selected model for image input.
    Yields raw text and Markdown-formatted text.
    """
    if model_name == "olmOCR-2-7B-1025":
        processor = processor_m
        model = model_m
    elif model_name == "Nanonets-OCR2-3B":
        processor = processor_x
        model = model_x
    elif model_name == "Chandra-OCR":
        processor = processor_v
        model = model_v
    elif model_name == "Dots.OCR":
        processor = processor_d
        model = model_d
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[prompt_full],
        images=[image],
        return_tensors="pt",
        padding=True).to(device)

    streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

image_examples = [
    ["OCR the content perfectly.", "examples/3.jpg"],
    ["Perform OCR on the image.", "examples/1.jpg"],
    ["Extract the contents. [page].", "examples/2.jpg"],
]

with gr.Blocks(css=css, theme=steel_blue_theme) as demo:
    gr.Markdown("# **Multimodal OCR3**", elem_id="main-title")
    with gr.Row():
        with gr.Column(scale=2):
            image_query = gr.Textbox(label="Query Input", placeholder="Enter your query here...")
            image_upload = gr.Image(type="pil", label="Upload Image", height=290)

            image_submit = gr.Button("Submit", variant="primary")
            gr.Examples(
                examples=image_examples,
                inputs=[image_query, image_upload]
            )
            
            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens", minimum=1, maximum=MAX_MAX_NEW_TOKENS, step=1, value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=4.0, step=0.1, value=0.7)
                top_p = gr.Slider(label="Top-p (nucleus sampling)", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                top_k = gr.Slider(label="Top-k", minimum=1, maximum=1000, step=1, value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty", minimum=1.0, maximum=2.0, step=0.05, value=1.1)
                
        with gr.Column(scale=3):
                gr.Markdown("## Output", elem_id="output-title")
                output = gr.Textbox(label="Raw Output Stream", interactive=False, lines=11, show_copy_button=True)
                with gr.Accordion("(Result.md)", open=False):
                    markdown_output = gr.Markdown(label="(Result.Md)")

                model_choice = gr.Radio(
                    choices=["Nanonets-OCR2-3B", "Chandra-OCR", "Dots.OCR", "olmOCR-2-7B-1025"],
                    label="Select Model",
                    value="Nanonets-OCR2-3B"
                )

    image_submit.click(
        fn=generate_image,
        inputs=[model_choice, image_query, image_upload, max_new_tokens, temperature, top_p, top_k, repetition_penalty],
        outputs=[output, markdown_output]
    )

if __name__ == "__main__":
    demo.queue(max_size=50).launch(mcp_server=True, ssr_mode=False, show_error=True)
