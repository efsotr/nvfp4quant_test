import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:200]")

MODEL_ID = "Llama-3.2-1B-Instruct"
SAVE_DIR = "Llama-3.2-1B-Instruct-NVFP4"

MAX_SEQUENCE_LENGTH = 2048
NUM_CALIBRATION_SAMPLES = 20

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def preprocess(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

ds = ds.map(preprocess)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="NVFP4",
    ignore=["lm_head"],
)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)