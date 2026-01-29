#!/usr/bin/env python3
import argparse
import time
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================
# Hooks
# ============================

def detector_hook(module, inp, out):
    # Only detection, no modification
    if torch.any(torch.abs(out) > 1e30):
        pass
    return out


def correction_hook(module, inp, out):
    # Detection + correction
    mask = torch.abs(out) > 1e30
    if mask.any():
        out = out.clone()
        out[mask] = 0.0
    return out


# ============================
# Register hooks
# ============================

def add_attention_hooks(model):
    for name, module in model.named_modules():
        if any(x in name for x in [
            "attention.self.query",
            "attention.self.key",
            "attention.self.value",
            "attention.output.dense"
        ]):
            module.register_forward_hook(correction_hook)


def add_ffn_hooks(model):
    for name, module in model.named_modules():
        if any(x in name for x in [
            "intermediate.dense",
            "output.dense",
            "mlp.c_fc",
            "mlp.c_proj"
        ]):
            module.register_forward_hook(correction_hook)


# ============================
# Measurement
# ============================

def measure_forward(model, inputs, runs):
    torch.set_grad_enabled(False)

    # Warmup
    for _ in range(10):
        _ = model(**inputs)

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(**inputs)
    end = time.perf_counter()

    return (end - start) / runs


# ============================
# Main
# ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--protection", type=str,
                        choices=["none", "attention+ffn"],
                        required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    # Force CPU
    device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

    # Fix for GPT-2 (no pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    text = "This is a test sentence for overhead measurement."
    inputs = tokenizer(text, return_tensors="pt",
                       padding="max_length", truncation=True,
                       max_length=64).to(device)

    # Apply protection mode
    # Apply protection mode
    if args.protection == "attention+ffn":
        add_attention_hooks(model)
        add_ffn_hooks(model)

    # Measure latency
    latency = measure_forward(model, inputs, args.iterations)
    print(f"Latency: {latency*1000:.3f} ms")


if __name__ == "__main__":
    main()
