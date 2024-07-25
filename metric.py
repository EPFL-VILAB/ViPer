# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on IDEFICS code bases
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics2
# --------------------------------------------------------

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForVision2Seq
from peft import PeftModel

def set_device(device='cuda:0'):
    torch.cuda.set_device(device)
    return device

device = DEVICE = set_device("cuda:0")

def load_context_images(negative_image_paths, positive_image_paths):
    context_images = []
    for i in range(len(negative_image_paths)):
        context_images.append(Image.open(negative_image_paths[i]))
        context_images.append(Image.open(positive_image_paths[i]))

    return context_images

def initialize_processor_and_model(device):
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        size={"longest_edge": 448, "shortest_edge": 378},
        do_image_splitting=False
    )

    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/idefics2-8b").to(device)
    model = PeftModel.from_pretrained(
        model, "EPFL-VILAB/Metric-ViPer"
    ).to(device)

    return processor, model

def calculate_score(processor, model, context_images, query_image):
    prompt = ""

    for i in range(len(context_images) // 2):
        prompt = prompt + "User:<image>Score for this image?<end_of_utterance>\n"
        prompt = prompt + "Assistant: -<end_of_utterance>\n"
        prompt = prompt + "User:<image>Score for this image?<end_of_utterance>\n"
        prompt = prompt + "Assistant: +<end_of_utterance>\n"

    context_images.append(Image.open(query_image))

    prompt = prompt + "User:<image>Score for this image?<end_of_utterance>\n"
    prompt = prompt + "Assistant: "

    inputs = processor(text=prompt, images=context_images, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
            outputs = model(**inputs)

    logits = outputs.get("logits")
    score = torch.exp(logits[:, -1][:, 648]) / (torch.exp(logits[:, -1][:, 648]) + torch.exp(logits[:, -1][:, 387]))
    score = score.item()

    return score

