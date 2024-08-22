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
    "metric_examples/disliked/0.png",
    "metric_examples/disliked/1.png",
    "metric_examples/disliked/2.png",
    "metric_examples/disliked/3.png",
    "metric_examples/disliked/4.png",
    "metric_examples/disliked/5.png",
    "metric_examples/disliked/6.png",
    "metric_examples/disliked/7.png",
]

positive_image_paths = [
    "metric_examples/liked/0.png",
    "metric_examples/liked/1.png",
    "metric_examples/liked/2.png",
    "metric_examples/liked/3.png",
    "metric_examples/liked/4.png",
    "metric_examples/liked/5.png",
    "metric_examples/liked/6.png",
    "metric_examples/liked/7.png",
]

query_image = "metric_examples/query3.png"

device = set_device("cuda:0")
context_images = load_context_images(negative_image_paths, positive_image_paths)
processor, model = initialize_processor_and_model(device)

score = calculate_score(processor, model, context_images, query_image)

print(score)
