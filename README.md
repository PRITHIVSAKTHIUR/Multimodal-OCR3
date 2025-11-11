# **Multimodal-OCR3**

Multimodal-OCR3 is an advanced Optical Character Recognition (OCR) application that leverages multiple state-of-the-art multimodal models to extract text from images. Built with a user-friendly Gradio interface, it supports models like Nanonets-OCR2-3B, Chandra-OCR, olmOCR-2-7B-1025, and Dots.OCR, enabling robust text extraction with customizable generation parameters.

This project is licensed under the [Apache License 2.0](LICENSE).

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

<img width="1918" height="1400" alt="1" src="https://github.com/user-attachments/assets/2177be87-e0a3-4148-8487-5215e026d192" />
<img width="1918" height="1784" alt="2" src="https://github.com/user-attachments/assets/fa43f738-a15d-4919-9efa-a6ec8e07d73b" />

---

## Features
- **Multiple OCR Models**: Supports four OCR models: Nanonets-OCR2-3B, Chandra-OCR, olmOCR-2-7B-1025, and Dots.OCR.
- **Gradio Interface**: Intuitive web-based UI for uploading images and entering queries.
- **Customizable Parameters**: Adjust max new tokens, temperature, top-p, top-k, and repetition penalty for text generation.
- **Real-time Streaming**: View OCR output as it is generated.
- **Example Inputs**: Predefined example queries and images for quick testing.
- **Custom Theme**: Styled with a unique SteelBlue theme for an enhanced user experience.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-OCR3.git
   cd Multimodal-OCR3
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have Python 3.10+ installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes dependencies such as `torch`, `transformers`, `gradio`, and `flash-attn`. See the [Requirements](#requirements) section for the full list.

4. **Download Models**:
   The application automatically downloads and caches the required models from Hugging Face during the first run. Ensure you have sufficient disk space in the `./model_cache` directory.

5. **Run the Application**:
   Launch the Gradio interface:
   ```bash
   python app.py
   ```
   This will start a local web server, and you can access the interface via the provided URL (typically `http://localhost:7860`).

## Usage

1. **Access the Interface**:
   Open the Gradio interface in your browser after running the application.

2. **Select a Model**:
   Choose one of the supported models (e.g., Nanonets-OCR2-3B) from the radio buttons.

3. **Upload an Image**:
   Upload an image containing text you want to extract.

4. **Enter a Query**:
   Provide a query (e.g., "Perform OCR on the image") in the text input box.

5. **Adjust Advanced Options** (optional):
   Modify parameters like `max_new_tokens`, `temperature`, `top_p`, `top_k`, and `repetition_penalty` for fine-tuned results.

6. **Submit**:
   Click the "Submit" button to process the image and view the extracted text in real-time.

7. **View Output**:
   The raw text output and Markdown-formatted results will appear in the output section.

### Example Usage
The application includes example inputs for quick testing:
- **Query**: "Perform OCR on the image."
- **Image**: `examples/1.jpg`
- **Output**: Extracted text from the image.

## Model Details
Multimodal-OCR3 integrates the following models:
- **Nanonets-OCR2-3B**: A lightweight, efficient OCR model for text extraction.
- **Chandra-OCR**: A high-precision model optimized for complex documents.
- **olmOCR-2-7B-1025**: A robust model for diverse image types, developed by Allen AI.
- **Dots.OCR**: A custom-patched model for enhanced OCR performance.

All models are loaded with `torch.float16` or `torch.bfloat16` precision and utilize GPU acceleration (if available) via CUDA.

## Requirements
The following packages are required to run Multimodal-OCR3:
```
flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
transformers-stream-generator
huggingface_hub
qwen-vl-utils
sentencepiece
opencv-python
torch==2.6.0
transformers
torchvision
matplotlib
accelerate
requests
hf_xet
spaces
pillow
gradio
einops
peft
fpdf
timm
av
```

Install them using:
```bash
pip install -r requirements.txt
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please ensure your code adheres to the project's coding standards and includes appropriate documentation.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [Hugging Face](https://huggingface.co/) for providing the pretrained models.
- [Gradio](https://gradio.app/) for the intuitive web interface framework.
- [PyTorch](https://pytorch.org/) for the deep learning backend.
- The open-source community for contributions to dependencies like `transformers` and `flash-attn`.
