# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on Stable Diffusion and IDEFICS code bases
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics2
# --------------------------------------------------------

from ViPer import (
    set_device,
    load_images,
    initialize_processor_and_model,
    prepare_prompt_and_inputs,
    generate_texts,
    extract_features,
    initialize_pipelines,
    generate_images
)

def main():
    comments = [
        "These are beautiful, intricate patterns. Very elegant, and the teal blue colors are lovely. I love the flowing lines.",
        "I find the Pointillism pattern very interesting. I love it.",
        "I love the transparent and multi-layered texture here. The lines are flowing and it feels free yet structured. It has a gloomy, haunted, mysterious vibe to it. And the colors? Purples and indigo blues are perfection!",
        "The colors here don't quite work for me. They feel a bit unmatched and artificial. The concept also seems a bit boring and artificial to me.",
        "This piece feels too busy, and the colors don't seem to match at all. It feels a bit overwhelming to look at.",
        "I really like the cyberpunk vibes here! The neon lighting is pretty cool to look at, and the mysterious, creepy vibes are so cool!",
        "This piece doesn't resonate with me. I'm not a fan of naive art or vibrant colors. It feels lacking in skills to me.",
        "I'm a bit conflicted about this one. I like how everything here feels like an illusion and how extreme the patterns are. It's both unsettling and nice. I just wish it wasn't so abstract.",
        "Perfection! Such a beautiful combination of Art Nouveau and Steampunk! The dark gold, silver, and blue combination makes it feel so expensive. It feels like a mystical dream, and I love it.",
        "This one isn't for me. The vibrant colors are too bright and overwhelming. The 3D effect isn't quite working for me either. The bright yellow is a bit too intense and realistic.",
        "Gorgeous work! I love the dark Surrealism here! It feels so nightmarish and exciting! I do wish it had more colors though. A purple and blue palette would be awesome."
    ]

    image_paths = [
        "images/6.png",
        "images/9.png",
        "images/13.png",
        "images/16.png",
        "images/18.png",
        "images/21.png",
        "images/55.png",
        "images/39.png",
        "images/42.png",
        "images/45.png",
        "images/22.png",
    ]

    prompts = [
        "Town, People walking in the streets",
        "Whimsical tea party in a bioluminescent forest",
        "Tiny houses on top of each other above clouds",
        "A person reaching for stars",
        "Human in a frame",
        "Happy cats"
    ]

    output_dir = "results/"

    device = set_device("cuda:0")
    
    # Initialize processor, model and inputs
    images = load_images(image_paths)
    processor, model = initialize_processor_and_model(device)
    inputs = prepare_prompt_and_inputs(processor, images, comments)

    # Generate and extract vp
    generated_texts = generate_texts(processor, model, inputs)
    vp_pos, vp_neg = extract_features(generated_texts)

    # Initialize pipelines and generate images
    pipe, refiner = initialize_pipelines(device)
    generate_images(pipe, refiner, prompts, vp_pos, vp_neg, output_dir)

if __name__ == "__main__":
    main()
