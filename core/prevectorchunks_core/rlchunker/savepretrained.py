import torch, os

def save_pretrained_model(policy_model, output_dir="pretrained_chunker"):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(policy_model.state_dict(), os.path.join(output_dir, "policy_model.pt"))
    print(f"✅ Saved pretrained model to {output_dir}/policy_model.pt")

save_pretrained_model(policy, "rlchunker/pretrained")
