#!/usr/bin/env python3
"""
Demo script for the video processing pipeline
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pipeline import VideoToSilhouettePipeline
from inference.predictor import ScoliosisPredictor, EnsemblePredictor


def main():
    parser = argparse.ArgumentParser(description='Scoliosis Detection from Phone Video')
    parser.add_argument('video', help='Path to input video file')
    parser.add_argument('--model', default='scoliosis_app/experiments/improved_ScoNet/checkpoints/clean_weights_sconet.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-type', default='sconet', choices=['sconet', 'sconet_mt'],
                       help='Model type to use')
    parser.add_argument('--output', default='output/', help='Output directory')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
        
    print("="*60)
    print("SCOLIOSIS DETECTION - PHONE VIDEO ANALYSIS")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Model: {args.model_type}")
    print(f"Device: {args.device}")
    print("-"*60)
    
    try:
        # Initialize pipeline
        print("\n📹 Initializing video processing pipeline...")
        pipeline = VideoToSilhouettePipeline(args.config)
        
        # Process video
        print("\n🎬 Processing video to silhouettes...")
        results = pipeline.process_video(args.video, args.output)
        
        print(f"✅ Processed {results['num_frames_processed']} frames")
        print(f"✅ Valid detections: {results['num_valid_detections']}")
        print(f"✅ Silhouette shape: {results['sequence_shape']}")
        
        # Check if model checkpoint exists
        if not Path(args.model).exists():
            print(f"\n⚠️  Model checkpoint not found: {args.model}")
            print("Silhouettes have been generated and saved.")
            print(f"Location: {args.output}")
            return
            
        # Initialize predictor
        print("\n🧠 Loading trained model...")
        predictor = ScoliosisPredictor(
            model_path=args.model,
            model_type=args.model_type,
            device=args.device
        )
        
        # Run prediction
        print("\n🔍 Running inference...")
        prediction = predictor.predict(results['silhouettes'])
        
        # Display results
        print("\n" + "="*60)
        print("SCREENING RESULTS")
        print("="*60)
        
        print(predictor.explain_prediction(prediction))
        
        print("\n" + "-"*60)
        print("CLINICAL RECOMMENDATION")
        print("-"*60)
        print(predictor.get_risk_assessment(prediction))
        
        print("\n" + "="*60)
        print("✅ Analysis Complete")
        print(f"Results saved to: {args.output}")
        print("="*60)
        
        # Save prediction results
        import json
        prediction_path = Path(args.output) / f"{Path(args.video).stem}_prediction.json"
        with open(prediction_path, 'w') as f:
            json.dump(prediction, f, indent=2)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
