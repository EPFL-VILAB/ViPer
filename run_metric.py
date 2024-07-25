# Copyright (c) EPFL VILAB.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Based on IDEFICS code bases
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/idefics2
# --------------------------------------------------------

from metric import (
    set_device,
    load_context_images,
    initialize_processor_and_model,
    calculate_score
)

negative_image_paths = [
    "disliked/0.png",
    "disliked/1.png",
    "disliked/2.png",
    "disliked/3.png",
    "disliked/4.png",
    "disliked/5.png",
    "disliked/6.png",
    "disliked/7.png",
    "disliked/8.png",
]

positive_image_paths = [
    "liked/0.png",
    "liked/1.png",
    "liked/2.png",
    "liked/3.png",
    "liked/4.png",
    "liked/5.png",
    "liked/6.png",
    "liked/7.png",
    "liked/8.png",
]

query_image = "query.png"

device = set_device("cuda:0")
context_images = load_context_images(negative_image_paths, positive_image_paths)
processor, model = initialize_processor_and_model(device)

score = calculate_score(processor, model, context_images, query_image)

print(score)
