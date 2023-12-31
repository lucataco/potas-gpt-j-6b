# This file runs during container build time to get model weights built into the container

from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    MODEL_NAME = "EleutherAI/gpt-j-6b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True
    )

if __name__ == "__main__":
    download_model()