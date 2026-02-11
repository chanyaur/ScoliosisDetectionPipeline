# 📱 Phone Video Processing Pipeline for Scoliosis Detection

A complete pipeline for converting phone-recorded walking videos into silhouette sequences suitable for scoliosis prediction using trained ScoNet models.

## 🎯 Overview

This pipeline processes phone videos through the following stages:
1. **Video Loading** - Extract and sample frames from video
2. **Person Detection** - Detect and track the primary person using YOLOv8
3. **Segmentation** - Segment human from background using MediaPipe
4. **Silhouette Generation** - Convert to 64x64 binary silhouettes
5. **Model Inference** - Predict scoliosis classification using trained models

## 🚀 Quick Start

### Installation

```bash
cd video_pipeline
pip install -r requirements.txt
```

### Basic Usage

```python
from video_pipeline.pipeline import VideoToSilhouettePipeline
from video_pipeline.inference.predictor import ScoliosisPredictor

# Process video to silhouettes
pipeline = VideoToSilhouettePipeline("config.yaml")
results = pipeline.process_video("walking_video.mp4", "output/")

# Run inference
predictor = ScoliosisPredictor(
    model_path="../scoliosis_app/experiments/sconet_best.pth",
    model_type="sconet",
    device="cuda"
)

prediction = predictor.predict(results['silhouettes'])
print(f"Prediction: {prediction['predicted_label']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## 📁 Project Structure

```
video_pipeline/
├── core/                      # Core processing components
│   ├── video_loader.py       # Video loading and frame extraction
│   ├── person_detector.py    # YOLOv8 person detection
│   ├── person_tracker.py     # Simple object tracking
│   ├── segmentation.py       # Human segmentation (MediaPipe)
│   └── silhouette_generator.py # Silhouette generation
│
├── inference/                 # Model inference components
│   ├── model_loader.py       # Load trained models
│   ├── predictor.py          # Run predictions
│   └── ensemble.py           # Ensemble predictions
│
├── preprocessing/             # Additional preprocessing
│   ├── video_stabilizer.py  # Stabilize shaky videos
│   └── walking_detector.py  # Detect walking segments
│
├── pipeline.py               # Main pipeline orchestrator
├── config.yaml              # Configuration file
└── requirements.txt         # Dependencies
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
# Model paths
model:
  sconet_path: "../scoliosis_app/experiments/sconet_best.pth"
  sconet_mt_path: "../scoliosis_app/experiments/sconet_mt_best.pth"
  device: "cuda"  # or "cpu"

# Video processing
video:
  target_fps: 15
  min_duration: 5  # seconds
  max_duration: 60

# Detection settings
detection:
  model: "yolov8m.pt"
  confidence_threshold: 0.5

# Segmentation
segmentation:
  backend: "mediapipe"  # or "background_subtraction"
  confidence_threshold: 0.7

# Output
silhouette:
  output_size: [64, 64]
  sequence_length: 300
```

## 📊 Complete End-to-End Example

```python
import numpy as np
from video_pipeline.pipeline import VideoToSilhouettePipeline
from video_pipeline.inference.predictor import ScoliosisPredictor

# Initialize pipeline
pipeline = VideoToSilhouettePipeline("config.yaml")

# Process phone video
video_path = "patient_walking.mp4"
try:
    # Extract silhouettes
    results = pipeline.process_video(video_path, "output/")
    
    # Initialize predictor with trained model
    predictor = ScoliosisPredictor(
        model_path="../scoliosis_app/experiments/sconet_best.pth",
        model_type="sconet",
        device="cuda"
    )
    
    # Get prediction
    prediction = predictor.predict(results['silhouettes'])
    
    # Display results
    print("\n" + "="*50)
    print("SCOLIOSIS SCREENING RESULTS")
    print("="*50)
    print(f"Video: {video_path}")
    print(f"Frames processed: {results['num_frames_processed']}")
    print(f"Valid detections: {results['num_valid_detections']}")
    print("\n" + predictor.explain_prediction(prediction))
    print("\n" + predictor.get_risk_assessment(prediction))
    
except Exception as e:
    print(f"Error processing video: {e}")
```

## 🎨 Visualization

The pipeline automatically generates visualizations:
- Silhouette sequence preview
- Detection bounding boxes
- Segmentation masks

## ⚙️ Advanced Features

### Ensemble Prediction

Use both ScoNet and ScoNet-MT for improved accuracy:

```python
from video_pipeline.inference.predictor import EnsemblePredictor

ensemble = EnsemblePredictor(
    sconet_path="../scoliosis_app/experiments/sconet_best.pth",
    sconet_mt_path="../scoliosis_app/experiments/sconet_mt_best.pth",
    device="cuda"
)

prediction = ensemble.predict(silhouettes)
```

### Batch Processing

Process multiple videos:

```python
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = pipeline.process_batch(video_paths, "output/")
```

### Custom Preprocessing

Add custom preprocessing steps:

```python
from video_pipeline.core.video_loader import VideoLoader

loader = VideoLoader(target_fps=30)
frames, metadata = loader.load_video("video.mp4")

# Apply custom preprocessing
frames = your_custom_preprocessing(frames)
```

## 🏥 Clinical Interpretation

### Classification Categories:
- **Positive (Scoliosis)**: Cobb angle > 10°
- **Neutral (Borderline)**: Monitoring required
- **Negative (Healthy)**: No significant curvature

### Risk Assessment:
- **HIGH RISK**: Immediate medical evaluation recommended
- **MODERATE RISK**: Further screening recommended
- **BORDERLINE**: Monitor, rescreen in 6 months
- **LOW RISK**: No immediate concern

## 📈 Performance Metrics

Based on training results:
- **Sensitivity**: >95% (critical for screening)
- **Specificity**: 70-80%
- **Overall Accuracy**: 80-85%

## 🐛 Troubleshooting

### Common Issues:

1. **No person detected**
   - Ensure person is clearly visible
   - Check lighting conditions
   - Person should be walking perpendicular to camera

2. **Poor segmentation**
   - Try different segmentation backend
   - Adjust confidence threshold
   - Ensure plain background

3. **Model loading error**
   - Check model path in config
   - Ensure checkpoint file exists
   - Verify CUDA availability if using GPU

## 📝 Requirements

### Hardware:
- CPU: Multi-core processor recommended
- GPU: NVIDIA GPU with CUDA support (optional, for faster inference)
- RAM: Minimum 8GB

### Software:
- Python 3.8+
- OpenCV 4.8+
- PyTorch 2.0+
- MediaPipe 0.10+

## 🔬 Research Context

This pipeline implements the video processing component for the research paper:
"Scoliosis1K: A Large-Scale Dataset for Video-Based Scoliosis Screening"

Key innovations:
- Non-invasive screening using phone videos
- Gait-based scoliosis detection
- Deep learning with temporal modeling

## 📄 License

This project is for research purposes. For clinical deployment, appropriate medical device regulations must be followed.

## 🙏 Acknowledgments

- ScoNet architecture from Scoliosis1K paper
- YOLOv8 for person detection
- MediaPipe for human segmentation
- OpenGait framework for gait recognition

## 📮 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the configuration file
3. Ensure all dependencies are installed
4. Check model checkpoint paths

---

**Note**: This is a screening tool and should not replace professional medical diagnosis. Always consult healthcare professionals for clinical decisions.
