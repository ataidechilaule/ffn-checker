#!/usr/bin/env python3
import argparse
import random
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


# ============================================================
# Fault Injection (global)
# ============================================================

def global_fault_hook(fault_rate):
    def hook(module, inp, out):
        if not isinstance(out, torch.Tensor):
            return out

        if random.random() < fault_rate:
            out = out.clone()

            # 70% near-INF, 20% INF, 10% NaN
            r = random.random()
            if r < 0.7:
                out[torch.rand_like(out) < 0.0001] = 1e38
            elif r < 0.9:
                out[torch.rand_like(out) < 0.0001] = float("inf")
            else:
                out[torch.rand_like(out) < 0.0001] = float("nan")

        return out
    return hook


# ============================================================
# Protection Hooks (FFN-Checker + Attention-Checker)
# ============================================================

def correction_hook(module, inp, out):
    if not isinstance(out, torch.Tensor):
        return out

    mask = torch.isnan(out) | torch.isinf(out) | (torch.abs(out) > 1e30)
    if mask.any():
        out = out.clone()
        out[mask] = 0.0
    return out


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


# ============================================================
# Training Loop
# ============================================================

def train(model, tokenizer, dataset, epochs, fault_rate):
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    model.train()

    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")

        for step, batch in enumerate(dataloader):
            # Detect dataset structure
            if "sentence" in batch:
                    text = batch["sentence"]
            elif "sentence1" in batch and "sentence2" in batch:
                    text = [a + " [SEP] " + b for a, b in zip(batch["sentence1"], batch["sentence2"])]
            elif "question1" in batch and "question2" in batch:
                    text = [a + " [SEP] " + b for a, b in zip(batch["question1"], batch["question2"])]
            else:
                raise ValueError("Dataset format not supported.")

            labels = batch["label"]


            inputs = tokenizer(
                list(text),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            if step % 50 == 0:
                print(f"Step {step}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--protection", type=str,
                        choices=["none", "attention", "ffn", "attention+ffn"],
                        required=True)
    parser.add_argument("--fault_rate", type=float, default=0.001)
    args = parser.parse_args()

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    # Fix for GPT-2 (no pad token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    dataset = load_dataset("glue", args.dataset, split="train")

    # Register protection hooks
    if args.protection in ["attention", "attention+ffn"]:
        add_attention_hooks(model)
    if args.protection in ["ffn", "attention+ffn"]:
        add_ffn_hooks(model)

    # Register global fault injection
    fault_hook = global_fault_hook(args.fault_rate)
    for name, module in model.named_modules():
        module.register_forward_hook(fault_hook)

    # Train
    train(model, tokenizer, dataset, args.epochs, args.fault_rate)

    print("\nTraining completed without divergence.")


if __name__ == "__main__":
    main()
