import argparse
import torch
from transformers import AutoModelForSequenceClassification

def mark_attention(model):
    for name, module in model.named_modules():
        if any(x in name for x in ["attention.self.query", "attention.self.key", "attention.self.value"]):
            module._protected = True

def mark_ffn(model):
    for name, module in model.named_modules():
        if any(x in name for x in ["intermediate.dense", "output.dense", "mlp.c_fc", "mlp.c_proj"]):
            module._protected = True

def count_ffn_ops(model):
    total = 0
    protected = 0

    for name, module in model.named_modules():
        if any(x in name for x in ["intermediate.dense", "output.dense", "mlp.c_fc", "mlp.c_proj"]):
            total += 1
            if hasattr(module, "_protected") and module._protected:
                protected += 1

    return total, protected

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--protection", type=str, choices=["attention", "attention+ffn"])
    args = parser.parse_args()

    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    # Reset protection flags
    for _, module in model.named_modules():
        module._protected = False

    # Apply protection mode
    if args.protection == "attention":
        mark_attention(model)
    elif args.protection == "attention+ffn":
        mark_attention(model)
        mark_ffn(model)

    total, protected = count_ffn_ops(model)

    print("=== FFN Coverage Test ===")
    print(f"Total FFN operations: {total}")
    print(f"Protected operations: {protected}")
    print(f"Coverage: {protected / total * 100:.2f}%")

if __name__ == "__main__":
    main()
