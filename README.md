# **Multimodal-OCR3**

Multimodal-OCR3 is a highly capable, experimental optical character recognition and visual processing suite designed for precise text extraction, document parsing, and markdown generation. Leveraging a powerful selection of vision-language and causal language models—including architectures like Nanonets-OCR2-3B, Chandra-OCR, Dots.OCR, and olmOCR-2-7B—this application specializes in deciphering complex document layouts, dense texts, and real-world scene imagery. The tool features a highly customized, interactive web interface that enables users to effortlessly upload screenshots, receipts, and multi-page documents for rapid analysis. With built-in support for fully GPU-accelerated inference and granular manipulation of generation parameters, Multimodal-OCR3 provides researchers and developers with a streamlined environment for building, testing, and deploying robust document intelligence and multimodal workflows.

<img width="1920" height="1798" alt="Screenshot 2026-03-22 at 12-34-40 Multimodal OCR3 - a Hugging Face Space by prithivMLmods" src="https://github.com/user-attachments/assets/5ac37b79-b5af-484c-a77c-3076a1d1cba0" />

### **Key Features**

* **Multi-Model Architecture:** Seamlessly switch between specialized models directly from the interface. Supported models include `Nanonets-OCR2-3B`, `Chandra-OCR`, `Dots.OCR`, and `olmOCR-2-7B-1025`.
* **Custom User Interface:** Features a bespoke, responsive Gradio frontend built with custom HTML, CSS, and JavaScript. It includes a drag-and-drop media zone, real-time output streaming, and an integrated advanced settings panel.
* **Granular Inference Controls:** Fine-tune the AI's output by adjusting text generation parameters such as Maximum New Tokens, Temperature, Top-p, Top-k, and Repetition Penalty.
* **Output Management:** Built-in actions allow users to instantly copy the raw output text to their clipboard or save the generated response directly as a `.txt` file.
* **Flash Attention 2 Integration:** Utilizes `kernels-community/flash-attn2` for optimized, memory-efficient inference on compatible GPUs.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── 4.jpg
├── app.py
├── LICENSE
├── pre-requirements.txt
├── README.md
└── requirements.txt
```

### **Installation and Requirements**

To run Multimodal-OCR3 locally, you need to configure a Python environment with the following dependencies. Ensure you have a compatible CUDA-enabled GPU for optimal performance.

**1. Install Pre-requirements**
Run the following command to update pip to the required version:
```bash
pip install pip>=23.0.0
```

**2. Install Core Requirements**
Install the necessary machine learning and UI libraries. You can place these in a `requirements.txt` file and run `pip install -r requirements.txt`.

```text
git+https://github.com/huggingface/transformers.git@v4.57.6
git+https://github.com/huggingface/accelerate.git
git+https://github.com/huggingface/peft.git
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.8.0
torchvision
matplotlib
requests
kernels
hf_xet
spaces
pillow
gradio
av
```

### **Usage**

Once your environment is set up and the dependencies are installed, you can launch the application by running the main Python script:

```bash
python app.py
```

After the script initializes the interface, it will provide a local web address (usually `http://127.0.0.1:7860/`) which you can open in your browser to interact with the models. Note that the selected models will be downloaded and loaded into VRAM upon their first invocation.

### **License and Source**

* **License:** Apache License - Version 2.0
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR3.git](https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR3.git)
