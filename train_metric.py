import os
import random
import pandas as pd
import torch
from PIL import Image
from collections import defaultdict
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig

# Paths and dataset sizes
csv_path = 'images.csv'
root_path = '/images/'
train_set_size = 12000
test_set_size = 1000

# Device and configuration flags
DEVICE = "cuda:0"
USE_LORA = False
USE_QLORA = True

# Load processor and model configuration
processor = AutoProcessor.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    do_image_splitting=False,
    size={"longest_edge": 448, "shortest_edge": 378}
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    use_dora=False if USE_QLORA else True,
    init_lora_weights="gaussian"
)

if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

model = Idefics2ForConditionalGeneration.from_pretrained(
    "HuggingFaceM4/idefics2-8b",
    torch_dtype=torch.float16,
    quantization_config=bnb_config if USE_QLORA else None,
)
model.add_adapter(lora_config)
model.enable_adapters()

# Load dataset
dataframe = pd.read_csv(csv_path, encoding='utf8').iloc[::-1]

# Create train and test sets
def create_dataset(dataframe, set_size, offset=0):
    dataset = defaultdict(dict)
    for index, row in dataframe.iterrows():
        if index < set_size + offset:
            dataset[index] = [
                (row[f'Personalized_Image_{j+1}'], dataframe.iloc[random.randint(0, set_size - 1)][f'Personalized_Image_{j+1}'])
                for j in range(10)
            ]
    return dataset

train_set = create_dataset(dataframe, train_set_size)
test_set = create_dataset(dataframe, test_set_size, offset=train_set_size)

# Data Collator class
class MyDataCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __call__(self, examples):
        texts, images, targets = [], [], []

        for example in examples:
            curr_imgs, messages = [], []

            for negative_image, positive_image in example[:-1]:
                negative_image = os.path.join(root_path, negative_image)
                positive_image = os.path.join(root_path, positive_image)

                if not os.path.isfile(negative_image) or not os.path.isfile(positive_image):
                    print("Warning: image not found.")
                    continue

                order = random.randint(0, 1)
                imgs = [negative_image, positive_image] if order % 2 == 0 else [positive_image, negative_image]
                for img, score in zip(imgs, ["-", "+"]):
                    curr_imgs.append(Image.open(img))
                    messages.append(
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Score for this image?"}]}
                    )
                    messages.append(
                        {"role": "assistant", "content": [{"type": "text", "text": score}]}
                    )

            text = processor.apply_chat_template(messages, add_generation_prompt=False)[:-19]
            texts.extend([text.strip()] * 2)
            targets.extend(["-", "+"])
            images.extend([curr_imgs] * 2)

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        targets = processor(text=targets, return_tensors="pt", padding=True)
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][batch["labels"] == processor.tokenizer.pad_token_id] = self.image_token_id
        batch["targets"] = targets["input_ids"][:, -1].reshape(2, 1)

        return batch

data_collator = MyDataCollator(processor)

# Training arguments
training_args = TrainingArguments(
    num_train_epochs=2000,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=0,
    learning_rate=1e-5,
    weight_decay=0.01,
    logging_steps=1,
    output_dir="/proxy_metric",
    save_strategy="steps",
    save_steps=20,
    evaluation_strategy="steps",
    eval_steps=20,
    fp16=True,
    remove_unused_columns=False,
    report_to="wandb",
    do_eval=True
)

# Custom Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        def process_inputs(inputs, idx):
            sub_inputs = {key: val[idx:idx+1] for key, val in inputs.items()}
            targets = sub_inputs.pop("targets").reshape(1)
            logits = model(**sub_inputs).get("logits")
            return logits, targets
        
        negative_logits, negative_targets = process_inputs(inputs, 0)
        positive_logits, positive_targets = process_inputs(inputs, 1)

        loss_fct = torch.nn.CrossEntropyLoss().to(DEVICE)
        loss = (loss_fct(negative_logits[:, -1], negative_targets) + 
                loss_fct(positive_logits[:, -1], positive_targets)) / 2

        torch.cuda.empty_cache()
        return (loss, negative_logits) if return_outputs else loss

# Training
trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=test_set
)

trainer.train()
