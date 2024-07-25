# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on Stable Diffusion and IDEFICS code bases
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics2
# --------------------------------------------------------


import random
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoModelForVision2Seq

import sys
import importlib

class CustomLoader(importlib.machinery.SourceFileLoader):
    def get_filename(self, fullname):
        return self.path

class CustomFinder(importlib.machinery.PathFinder):
    def __init__(self, custom_path, module_name):
        self.custom_path = custom_path
        self.module_name = module_name

    def find_spec(self, fullname, path, target=None):
        if fullname == self.module_name:
            loader = CustomLoader(fullname, self.custom_path)
            return importlib.util.spec_from_file_location(fullname, self.custom_path, loader=loader)
        return None


custom_file_path = 'pipeline_stable_diffusion_xl.py'
module_name = 'diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl'

# Add the custom importer to sys.meta_path
sys.meta_path.insert(0, CustomFinder(custom_file_path, module_name))

import diffusers
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

def randomize_vp(input_string):
    elements = [element.strip() for element in input_string.split(", ")]
    random.shuffle(elements)
    return ", ".join(elements)

def set_device(device='cuda:0'):
    torch.cuda.set_device(device)
    return device

device = DEVICE = set_device("cuda:0")

def load_images(image_paths):
    return [Image.open(path) for path in image_paths]

def initialize_processor_and_model(device):
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        size={"longest_edge": 448, "shortest_edge": 378},
        do_image_splitting=False
    )

    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b").to(device)
    model = PeftModel.from_pretrained(
        model, "EPFL-VILAB/VPE-ViPer"
    ).to(device)

    return processor, model

def prepare_prompt_and_inputs(processor, images, comments):
    prompt_template = """I will provide a set of artworks along with accompanying comments from a person. Analyze these artworks and the comments on them and identify artistic features such as present or mentioned colors, style, composition, mood, medium, texture, brushwork, lighting, shadow effects, perspective, and other noteworthy elements.

    Your task is to extract the artistic features the person likes and dislikes based on both the artworks' features and the person's comments. Focus solely on artistic aspects and refrain from considering subject matter.

    If the person expresses a preference for a specific aspect without clearly stating its category (e.g., appreciating the colors without specifying which colors), identify these specific features from the images directly to make the person's preference understandable without needing to see the artwork.

    Your output should consist of two concise lists of keywords: one listing the specific art features the person likes and another listing the specific features they dislike (specified in keyword format without using sentences).

    Here are the images and their corresponding comments:
    """

    content = [{"type": "text", "text": prompt_template}]
    for index, comment in enumerate(comments):
        content.append({"type": "image"})
        content.append({"type": "text", "text": f"Comment {index + 1}: {comment}"})

    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    return inputs

def generate_texts(processor, model, inputs):
    generated_ids = model.generate(
        **inputs, max_new_tokens=2000, repetition_penalty=0.99, do_sample=False
    )
    return processor.batch_decode(generated_ids, skip_special_tokens=True)

def extract_features(generated_texts):
    VPs = generated_texts[0].split("\nAssistant:")[-1].strip()
    liked_prefix = "Liked Art Features:"
    disliked_prefix = "Disliked Art Features:"

    liked_start = VPs.find(liked_prefix) + len(liked_prefix)
    disliked_start = VPs.find(disliked_prefix) + len(disliked_prefix)

    liked_features = VPs[liked_start:VPs.find(disliked_prefix)].strip()
    disliked_features = VPs[disliked_start:].strip()

    liked_features_set = set(liked_features.split(", "))
    disliked_features_set = set(disliked_features.split(", "))

    common_features = liked_features_set & disliked_features_set

    liked_features_unique = liked_features_set - common_features
    disliked_features_unique = disliked_features_set - common_features

    vp_pos = ", ".join(liked_features_unique)
    vp_neg = ", ".join(disliked_features_unique)

    return vp_pos, vp_neg

def initialize_pipelines(device):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16
    ).to(device)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16"
    ).to(device)

    return pipe, refiner

def generate_images(pipe, refiner, prompts, vp_pos, vp_neg, output_dir, width=1024, height=1024):
    for prompt in prompts:
        vp_pos = randomize_vp(vp_pos)
        vp_neg = randomize_vp(vp_neg)
        image = pipe(
            prompt=prompt, 
            width=width, 
            height=height, 
            num_inference_steps=40, 
            vp_pos=vp_pos, 
            vp_neg=vp_neg, 
            denoising_end=0.8, 
            output_type="latent"
        ).images
        image = refiner(
            prompt=prompt,
            num_inference_steps=40,
            denoising_start=0.8,
            image=image
        ).images[0]
        image.save(f"{output_dir}{prompt[:20]}.png")
