from potassium import Potassium, Request, Response

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    MODEL_NAME = "EleutherAI/gpt-j-6b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to("cuda")

    context = {
        "model": model,
        "tokenizer": tokenizer,
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    max_new = request.json.get("max_new_tokens", 6)
    model = context.get("model")
    tokenizer = context.get("tokenizer")

    # Tokenize
    input_tokens = tokenizer.encode(prompt,return_tensors="pt").to("cuda")
    output = model.generate(input_tokens, max_new_tokens=max_new, pad_token_id=50256)
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    return Response(
        json = {"output": output_text}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()