import argparse
import json
from pathlib import Path
import torch

from train_exclude_neutral import Trainer, get_config   # your Trainer + config
from models.sconet import create_sconet                 # to rebuild the exact model

def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved ScoNet checkpoint")
    parser.add_argument("--ckpt", required=False, help="Path to checkpoint .pth file")
    parser.add_argument("--save_json", action="store_true", help="Save results JSON next to the checkpoint")
    args = parser.parse_args()

    # ckpt_path = Path(args.ckpt)
    ckpt_path = Path("../../experiments/sconet_20251107_224331/checkpoints/latest.pth")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load checkpoint first so we can align configs (model_type, num_classes, etc.)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    ckpt_cfg = checkpoint.get("config", {})

    # Start from your default config, then align critical fields to the checkpoint
    cfg = get_config()
    for k in ["model_type", "num_classes", "n_frames", "data_root", "batch_size", "num_workers"]:
        if k in ckpt_cfg:
            cfg[k] = ckpt_cfg[k]

    # Build trainer, data, loss
    trainer = Trainer(cfg)
    trainer.setup_data()
    trainer.setup_loss_and_optimizer()

    # Rebuild the exact architecture that the checkpoint expects
    trainer.model = create_sconet(
        model_type=cfg["model_type"],
        num_classes=cfg["num_classes"],
        n_frames=cfg["n_frames"],
    ).to(trainer.device)

    # Load weights
    trainer.model.load_state_dict(checkpoint["model_state_dict"])
    epoch = int(checkpoint.get("epoch", -1))

    # Evaluate on your test loader
    print(f"\nEvaluating {ckpt_path.name} on test set...")
    test_loss, test_acc, test_sens, test_cm = trainer.validate(
        epoch, trainer.test_loader, phase="Test"
    )

    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Sensitivity (Positive): {test_sens:.4f}")
    print("Confusion Matrix:")
    print(test_cm)

    # Optional: save JSON next to the checkpoint
    if args.save_json:
        results = {
            "checkpoint": str(ckpt_path),
            "epoch": epoch,
            "test_accuracy": float(test_acc),
            "test_sensitivity": float(test_sens),
            "test_loss": float(test_loss),
            "confusion_matrix": test_cm.tolist(),
            "config": cfg,
        }
        out_json = ckpt_path.parent / "test_results_eval.json"
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to: {out_json}")

if __name__ == "__main__":
    main()
