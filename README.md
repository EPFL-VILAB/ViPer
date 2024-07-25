# ViPer: Visual Personalization of Generative Models via Individual Preference Learning

[Sogand Salehi](https://sogandstormesalehi.github.io/), [Mahdi Shafiei](), [Teresa Yeo](https://aserety.github.io/), [Roman Bachmann](https://roman-bachmann.github.io/), [Amir Zamir](https://vilab.epfl.ch/zamir/)


 [`Website`](https://viper.epfl.ch/) | [`arXiv`](https://arxiv.org/abs/2407.17365) | [`BibTeX`](#citation)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/EPFL-VILAB/ViPer)


Official implementation and pre-trained models for **ViPer: Visual Personalization of Generative Models via Individual Preference Learning**, ECCV 2024 poster & demo.


![main figure](./assets/pf-nightmode.png#gh-dark-mode-only)
![main figure](./assets/pf-lightmode.png#gh-light-mode-only)

We introduce **ViPer**, a method that personalizes the output of generative models to align with different users’ visual preferences for the same prompt. This is done via a one-time capture of the user’s general preferences and conditioning the generative model on them without the need for engineering detailed prompts. Notice how the results vary for the same prompt for different users based on their visual preferences in the above figure.


### Models

Our models are available on Huggingface. VPE is our fine-tuned vision-language model that extracts individual preferences from a set of images and a user's comments on them.

Our proxy metric is a model that, given a user's set of liked and disliked images of individuals and a query image, can predict a preference score for the query image, indicating how much the user would like it.

Visit [VPE](https://huggingface.co/EPFL-VILAB/VPE-ViPer) and [Proxy](https://huggingface.co/EPFL-VILAB/Metric-ViPer) to download these fine-tuned models. 

## Usage

### ViPer Personalized Generation

![method figure](./assets/viper-method-nightmode.png#gh-dark-mode-only)
![method figure](./assets/viper-method-lightmode.png#gh-light-mode-only)

The following code extracts a user's visual preferences from a set of images and the user's comments on them. These preferences are then used to guide Stable Diffusion in generating images that align with the individual's tastes.

The user should comment on images that evoke a **strong reaction**, whether negative or positive. The comments should explain why the user likes or dislikes an image from an artistic perspective. More detailed comments will generate more personalized results.

We recommend commenting on at least 8 images. Adjust the image paths and their corresponding comments lists. The personalized generations will be saved in the results/ directory.

```python
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

# Ensure that the order of the comments matches the path of the images they refer to.

image_paths = [
    "/images/6.png",
    "/images/9.png"
]

comments = [
    "These are beautiful, intricate patterns. Very elegant, and the teal blue colors are lovely. I love the flowing lines.",
    "The colors here don't quite work for me. They feel a bit unmatched and artificial. The concept also seems a bit boring and artificial to me.",
]

prompts = [
    "Whimsical tea party in a bioluminescent forest",
    "Tiny houses on top of each other above clouds"
]

output_dir = "results/"

device = set_device("cuda:0")
    
# Initialize processor, model and inputs
images = load_images(image_paths)
processor, model = initialize_processor_and_model(device)
inputs = prepare_prompt_and_inputs(processor, images, comments)

# Generate and extract the user's positive and negative visual preferences
generated_texts = generate_texts(processor, model, inputs)
vp_pos, vp_neg = extract_features(generated_texts)

# Initialize pipelines and generate images
pipe, refiner = initialize_pipelines(device)
generate_images(pipe, refiner, prompts, vp_pos, vp_neg, output_dir)

```

### ViPer Proxy Metric

![main figure](./assets/metric-nightmode.png#gh-dark-mode-only)
![main figure](./assets/metric-lightmode.png#gh-light-mode-only)


Our proxy metric is fine-tuned to predict a preference score for a given query image, based on a user's set of liked and disliked images. This score indicates how much the user would favor the query image.

We recommend using around 8 liked and 8 disliked images as context, ensuring that the number of liked and disliked images is equal. The query image's score ranges from 0 to 1, with higher scores indicating a higher preference for the image.

```python
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

# Specify the path of the query image
query_image = "query.png"

device = set_device("cuda:0")
    
# Initialize processor and model
device = set_device("cuda:0")
context_images = load_context_images(negative_image_paths, positive_image_paths)
processor, model = initialize_processor_and_model(device)

# Calculate and print score
score = calculate_score(processor, model, context_images, query_image)

print(score)

```


## Demo & visualizations

For more examples and interactive demos, please see our [`website`](https://viper.epfl.ch/) and [`Hugging Face Space`](https://huggingface.co/spaces/EPFL-VILAB/ViPer).


## License

Licensed under the Apache License, Version 2.0. See [LICENSE](/LICENSE) for details.

## Citation

If you find this repository helpful, please consider citing our work:

```BibTeX
@misc{salehi2024vipervisualpersonalizationgenerative,
      title={ViPer: Visual Personalization of Generative Models via Individual Preference Learning}, 
      author={Sogand Salehi and Mahdi Shafiei and Teresa Yeo and Roman Bachmann and Amir Zamir},
      year={2024},
      eprint={2407.17365},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.17365}, 
}
```
