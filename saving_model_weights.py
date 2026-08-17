import torch

checkpoint = torch.load("scoliosis_app\experiments\sconet_20260811_235437\checkpoints\epoch_30_USE.pth", map_location="cpu", weights_only=False)

# If it's a dict checkpoint
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint  # already state_dict

torch.save(state_dict, "clean_weights_sconetMT-binary.pth")

print("Saved clean weights.")


# CHECK SCONET OR MT

# import torch

# checkpoint = torch.load(
#     "scoliosis_app\experiments\sconet_20260811_235437\checkpoints\epoch_30.pth",
#     map_location="cpu",
#     weights_only=False
# )

# state_dict = checkpoint["model_state_dict"]

# print("Number of parameters:", len(state_dict))

# print("\nANGLE REGRESSOR:")
# for key in state_dict:
#     if "angle" in key.lower():
#         print(key)

# print("\nALL KEYS:")
# for key in state_dict:
#     print(key)