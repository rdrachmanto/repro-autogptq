import time

from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, GPTQConfig
from auto_gptq import AutoGPTQForCausalLM

model_id = "./models/Llama3.1-8B-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
quantization_config = GPTQConfig(
    bits=4, 
    dataset="c4",
    tokenizer=tokenizer,
    exllama_config={"version":2}
)

start = time.time()
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    quantization_config=quantization_config
)
end = time.time()

print(f"Quantization done in {end - start} seconds")

model.to("cuda")

quant_path = "./models/Llama3.1-8B-4bit"
model.save_pretrained(quant_path)
print(f"Model saved to {quant_path}")

quantized_model = AutoModelForCausalLM.from_pretrained(
    quant_path,
    device_map="auto"
)
print("Quantized model loaded to gpu!")
print("----------- DONE ------------!")
