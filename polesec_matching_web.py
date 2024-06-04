import gradio as gr
from PIL import Image
import traceback
import re
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor, Idefics2ForConditionalGeneration, BitsAndBytesConfig
import subprocess
import gc

# There is an insulator prominently displayed in the image. Answer strictly with a number. How many discs are there in the insulator?

# Global variables to keep track of current model and name
current_model = None
current_tokenizer_or_processor = None
current_model_name = None

# Load model function
def load_model(model_name, device, dtype):
    global current_model, current_tokenizer_or_processor, current_model_name

    # Check if the model needs to be changed
    if current_model_name != model_name:
        # Clear the existing model from memory
        if current_model is not None:
            del current_model
            current_model = None
        if current_tokenizer_or_processor is not None:
            del current_tokenizer_or_processor
            current_tokenizer_or_processor = None
        gc.collect()
        torch.cuda.empty_cache()

        
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b",do_image_splitting=False)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # model = Idefics2ForConditionalGeneration.from_pretrained(model_paths[model_name], torch_dtype=torch.float16, quantization_config=bnb_config).to(device)
        model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", torch_dtype=dtype, trust_remote_code=True).to(device)
        if model_name == 'IDEFICS2-8B':
            model.load_adapter('./polesec_match_idefics')
            model.enable_adapters()
        current_model = model
        current_tokenizer_or_processor = processor
        current_model_name = model_name
    
    return current_model, current_tokenizer_or_processor


# Inference function for IDEFICS2-8B
def inference_idefics2(image, text, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p, num_beams, model, processor, device):
    if text == "" and not image:
        return "Please input a query and optionally image(s)."

    if text == "" and image:
        return "Please input a text query along the image(s)."

    resulting_messages = [{"role": "user", "content": [{"type": "image"}] + [{"type": "text", "text": text}]}]

    prompt = processor.apply_chat_template(resulting_messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_args = {"max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty}
    if decoding_strategy == "Beam Search":
        generation_args["do_sample"] = False
        generation_args["num_beams"] = num_beams
    elif decoding_strategy == "Top P Sampling":
        generation_args["temperature"] = temperature
        generation_args["do_sample"] = True
        generation_args["top_p"] = top_p

    generation_args.update(inputs)
    generated_ids = model.generate(**generation_args)
    generated_texts = processor.batch_decode(generated_ids[:, generation_args["input_ids"].size(1):], skip_special_tokens=True)
    return generated_texts[0]

# UI components
model_selector = gr.Radio(label="Select Model", choices=["No LORA", "IDEFICS2-8B"], value="MiniCPM-V 2.0")
image_input = gr.Image(label="Upload your Image", type="pil")
text_input = gr.Textbox(label="Prompt", value="In this image we have colored bounding boxes. Group the bounding boxes that cover the same part of the powerline towers.")
submit_btn = gr.Button("Submit")
output_text = gr.Textbox(label="Output")

# Sliders and other controls for parameters
temperature_slider = gr.Slider(minimum=0.0, maximum=5.0, value=0.4, step=0.1, label="Sampling temperature")
top_p_slider = gr.Slider(minimum=0.01, maximum=0.99, value=0.8, step=0.01, label="Top P")
max_new_tokens_slider = gr.Slider(minimum=8, maximum=1024, value=512, step=1, label="Max New Tokens")
repetition_penalty_slider = gr.Slider(minimum=0.01, maximum=5.0, value=1.2, step=0.01, label="Repetition Penalty")
num_beams_slider = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Beams")
decoding_strategy_radio = gr.Radio(choices=["Beam Search", "Top P Sampling"], value="Beam Search", label="Decoding Strategy")

# Load model and run inference based on selected model
def run_inference(model_name, image, text, temperature, max_new_tokens, repetition_penalty, top_p, num_beams, decoding_strategy):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model, tokenizer_or_processor = load_model(model_name, device, dtype)

    # Pad image to square with black
    if image is not None:
        width, height = image.size
        if width != height:
            new_size = max(width, height)
            new_image = Image.new("RGB", (new_size, new_size), (0, 0, 0))
            new_image.paste(image, ((new_size - width) // 2, (new_size - height) // 2))
            image = new_image
        image = image.resize((224, 224))
    return inference_idefics2(image, text, decoding_strategy, temperature, max_new_tokens, repetition_penalty, top_p, num_beams, model, tokenizer_or_processor, device)

# Define the layout and interactions
with gr.Blocks() as demo:
    with gr.Row():
        model_selector.render()
        image_input.render()
    with gr.Row():
        text_input.render()
    with gr.Row():
        decoding_strategy_radio.render()
        num_beams_slider.render()
    with gr.Row():
        temperature_slider.render()
        top_p_slider.render()
    with gr.Row():
        max_new_tokens_slider.render()
        repetition_penalty_slider.render()
    with gr.Row():
        submit_btn.render()
    with gr.Row():
        output_text.render()

    submit_btn.click(
        run_inference, 
        inputs=[model_selector, image_input, text_input, temperature_slider, max_new_tokens_slider, repetition_penalty_slider, top_p_slider, num_beams_slider, decoding_strategy_radio], 
        outputs=output_text
    )

# Launch the demo
demo.launch(debug=True)
