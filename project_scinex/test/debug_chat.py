from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "User: Hello! Can you introduce yourself?\nAssistant:"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# FIX: added pad_token_id — without it, open-ended sampling can produce
#      garbled tokens or trigger a HuggingFace warning on some models.
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
