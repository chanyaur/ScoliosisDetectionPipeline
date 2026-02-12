import torch

checkpoint = torch.load("scoliosis_app/experiments/improved_ScoNet/checkpoints/latest.pth", map_location="cpu")

# If it's a dict checkpoint
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint  # already state_dict

torch.save(state_dict, "clean_weights.pth")

print("Saved clean weights.")
