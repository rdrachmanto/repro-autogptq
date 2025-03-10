import time
import argparse
from dataclasses import dataclass
import os

from transformers import AutoModelForCausalLM, LlamaForCausalLM, AutoTokenizer, GPTQConfig
from auto_gptq import AutoGPTQForCausalLM


@dataclass
class RunnerConfig:
    model_path: str
    results_dir: str
    quantized_model_name: str
    bits: int
    dataset: str 


def main(rc: RunnerConfig):
    tokenizer = AutoTokenizer.from_pretrained(rc.model_path)
    quantization_config = GPTQConfig(
        bits=4, 
        dataset="c4",
        tokenizer=tokenizer,
        exllama_config={"version":2}
    )

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        rc.model_path, 
        device_map="auto", 
        quantization_config=quantization_config
    )
    end = time.time()

    print(f"Quantization done in {end - start} seconds")

    model.to("cuda")

    os.makedirs(rc.results_dir, exist_ok=True)

    quant_path = f"./{rc.results_dir}/{rc.quantized_model_name}"
    model.save_pretrained(quant_path)
    print(f"Model saved to {quant_path}")

    quantized_model = AutoModelForCausalLM.from_pretrained(
        quant_path,
        device_map="auto"
    )
    print("Quantized model loaded to gpu!")
    print("----------- DONE ------------!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("--model-path", "-mp", required=True)
    parser.add_argument("--results-dir", "-rd", required=True)
    parser.add_argument("--quantized-model-name", "-qmn", required=True)
    parser.add_argument("--dataset", "-d", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--bits", "-b", choices=[2, 4, 6, 8], default=4, type=int)

    args = parser.parse_args()

    runner_config = RunnerConfig(args.model_path, args.results_dir, args.quantized_model_name, args.bits, args.dataset)

    main(runner_config)
