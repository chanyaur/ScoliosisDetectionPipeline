import streamlit as st
# import tensorflow as tf
# import tensorflow_hub as hub
from PIL import Image
import numpy as np

# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from video_pipeline.pipeline import VideoToSilhouettePipeline
from video_pipeline.inference.predictor import ScoliosisPredictor

# Constants
SAVED_MODEL_PATH = "scoliosis_app/experiments/improved_ScoNet_binary/checkpoints/latest.pth"  #!
TITLE = 'Full Video Pipeline for Scoliosis Detection'
TITLE_PADDED = '&emsp;'*4 + TITLE
IM_CONSTANTS = {'LOGO': 'https://i.ibb.co/xhcsJ0d/duck-guy.png'}


# Main Streamlit app
def main():
    st.set_page_config(TITLE, page_icon=IM_CONSTANTS['LOGO'], layout='wide')
    st.title(TITLE_PADDED)
    
    if 'show_content' not in st.session_state:
        st.session_state.show_content = False  # Content is invisible by default
    if "video_button_label" not in st.session_state:
        st.session_state.video_button_label = "Show video"
    if "predicted_done" not in st.session_state:
        st.session_state.predicted_done = True
    if "sample_video" not in st.session_state:
        st.session_state.sample_video = None
    
    pipeline = VideoToSilhouettePipeline("video_pipeline/config.yaml")  # be careful if config acc loads or not
    
    # UI setup and information display
    _, col1, _, col2, _ = st.columns([2, 5, 1, 6, 2.5])

    with col1:
        st.write('''Welcome! This website is designed to be a video-based pipeline for early scoliosis detection 
                 that takes a video all the way from user input to model prediction.''')
        
        # st.markdown(
        # """
        # Video Requirements
        # - At least 15fps
        # - At least 5s duration
        # """
        # )
        
        st.markdown('''
            Record a video lasting at least **five seconds** of yourself walking towards your camera. 
            Some tips for best results are below:
            - Keep the **entire body** visible (head to feet)
            - Use a **well-lit** area with a clear background
            - Walk in a **straight line**''')
        st.markdown("It may be best to record your video on a path or hallway to allow adequate space for walking.")
        st.caption("Accepted formats: **MP4, MOV**; Requirements: **≥ 5s, ≥ 15fps**")
        
        st.header("FAQ")
        
        with st.expander("What should my gait video look like?"):
            st.markdown("""
            For the best results, your video should:
            
            - Be **at least 5 seconds long** and recorded at **15 FPS or higher**
            - Show your **entire body from head to toe**
            - Capture you **walking naturally at a normal pace**
            - Be filmed from the **side (profile view)**
            - Keep your **whole body visible throughout the video**
            - Use a **stationary camera** with minimal shaking
            - Have **good lighting** and a clear, uncluttered background
            - Avoid loose clothing that significantly obscures your body shape
            """)

            st.markdown("**Example gait video:**")
            sample_video_path = "scoliosis_app/sample_videos/demo.MOV"
            if os.path.exists(sample_video_path):
                st.video(sample_video_path)
            else:
                st.caption("Sample video unavailable.")
            
        with st.expander("What are the stages of the pipeline?"):
            st.markdown('''
            This pipeline processes phone videos through the following stages:
            1. **Video Loading** - Extract and sample frames from video
            2. **Person Detection** - Detect and track the primary person using YOLOv8
            3. **Segmentation** - Segment human from background using MediaPipe
            4. **Silhouette Generation** - Convert to 64x64 binary silhouettes
            5. **Model Inference** - Predict scoliosis classification using trained models
            ''')
        
        with st.expander("What are the differences between the four models?"):
            st.write('''
            The four models differ in both their **prediction task** and **architecture**.

            **ScoNet (3-class)** classifies gait into three scoliosis categories: Positive
            (>10° Cobb angle), Neutral (~10°), or Negative (<10°).

            **ScoNet-MT (3-class)** uses the same classification framework, but adds an
            auxiliary regression task that predicts Cobb angle. This multi-task approach
            is designed to help the model learn features related to scoliosis severity in
            addition to the classification itself.

            **Binary ScoNet** removes the borderline Neutral class and distinguishes only
            between Positive and Negative cases. Removing ambiguous cases near the 10°
            diagnostic threshold creates a simpler classification problem.

            **Binary ScoNet-MT** combines the binary classification task with the
            multi-task architecture, using both classification and Cobb-angle prediction.
            ''')

        with st.expander("How well do the models perform?"):
            st.write('''
            Model performance was evaluated on previously unseen test data using metrics
            including accuracy, sensitivity, F1 score, and area under the ROC curve (AUC).

            The binary models performed particularly well. **Binary ScoNet achieved an
            AUC of 0.950, 90.8% accuracy, and a 92.2% F1 score**, while **Binary ScoNet-MT
            achieved an AUC of 0.954, 89.7% accuracy, and a 91.7% F1 score**.

            The three-class models were evaluated on the more difficult task of separating
            Positive, Neutral, and Negative cases. **ScoNet achieved 80.4% accuracy and
            88.0% sensitivity**, while **ScoNet-MT achieved 80.0% accuracy and 93.3%
            sensitivity**.

            These results suggest that the models can identify patterns associated with
            scoliosis from gait silhouettes, while the stronger binary performance also
            shows the difficulty of distinguishing borderline Neutral cases.
            ''')
            
        with st.expander("How do I interpret my results?"):
            st.markdown(
                '''
                ### Classification Categories:
- **Positive (Scoliosis)**: Cobb angle > 10°
- **Neutral (Borderline)**: Monitoring required
- **Negative (Healthy)**: No significant curvature

### Risk Assessment:
- **HIGH RISK**: Immediate medical evaluation recommended
- **MODERATE RISK**: Further screening recommended
- **BORDERLINE**: Monitor, rescreen in 6 months
- **LOW RISK**: No immediate concern
                '''
            )
        
        # btn = st.button("reload col1")

    with col2:
        model_type = st.selectbox(label="Select preferred model", options=["ScoNet", "ScoNet-MT"]) # ["ScoNet", "ScoNet-MT", "ScoNet (binary)", "ScoNet-MT (binary)"])
        uploaded_file = st.file_uploader("Upload a gait video (.mp4, .mov)", type=["mp4", "mov"])

        st.markdown("**Don't have a gait video? Try a sample.**")
        if st.button("Try with a sample video"):
            with open("scoliosis_app/sample_videos/demo.MOV", "rb") as f:
                st.session_state.sample_video = f.read()
        # placeholder = st.empty()
        # placeholder.info("Results will appear here!")
        
        if uploaded_file is not None:
            video_bytes = uploaded_file.read()
                        
            run_pipeline_on_video(video_bytes, model_type, pipeline)
            btn = st.button(label=st.session_state.video_button_label, on_click=toggle_content_visibility)
            with st.empty():
                if st.session_state.show_content:
                    # TO SHOW THE VIDEO
                    st.video(video_bytes, autoplay=True, muted=True, width=180)
                
                # placeholder.info("Processing video...")

        elif st.session_state.sample_video is not None:
            video_bytes = st.session_state.sample_video

            run_pipeline_on_video(video_bytes, model_type, pipeline)
            btn = st.button(label=st.session_state.video_button_label, on_click=toggle_content_visibility)
            with st.empty():
                if st.session_state.show_content:
                    st.video(video_bytes, autoplay=True, muted=True, width=180)
                
    st.write(5*"\n")
    
            
        
def toggle_content_visibility():
    st.session_state.show_content = not st.session_state.show_content
    if st.session_state.show_content:
        st.session_state.video_button_label = "Hide video"
    else:
        st.session_state.video_button_label = "Show video"
    
    
@st.cache_data
def run_pipeline_on_video(video_bytes, model_type, _pipeline):
    # Save file
    with open("temp_video.mp4", "wb") as f:
        f.write(video_bytes)

    # Run your pipeline once
    results = _pipeline.process_video("temp_video.mp4")

    if(model_type == "ScoNet"):
        predictor = ScoliosisPredictor("scoliosis_app/experiments/improved_ScoNet/checkpoints/clean_weights_sconet.pth")  # modify w model type
    elif(model_type == "ScoNet-MT"):
        predictor = ScoliosisPredictor(model_path="scoliosis_app/experiments/improved_ScoNetMT/checkpoints/clean_weights_sconetMT.pth", model_type='sconet_mt')  # modify w model type
    # elif(model_type == "ScoNet (binary)"):
    #     # predictor = ScoliosisPredictor(model_path = "scoliosis_app/experiments/improved_ScoNet2/checkpoints/clean_weights_sconet-binary.pth", model_type="sconet_binary")  # name is wrong but actually sconetbinary2
    #     predictor = ScoliosisPredictor(model_path="scoliosis_app/experiments/final_models/clean_weights_sconetMT-binary.pth", model_type = "sconet_mt_binary")  # modify w model type
    # elif(model_type == "ScoNet-MT (binary)"):
    #     # predictor = ScoliosisPredictor(model_path="scoliosis_app/experiments/improved_ScoNetMT_binary2/checkpoints/clean_weights_sconetMT-binary.pth", model_type = "sconet_mt_binary")  # modify w model type
    #     predictor = ScoliosisPredictor(model_path="scoliosis_app/experiments/final_models/clean_weights_sconetMT-binary.pth", model_type = "sconet_mt_binary")  # modify w model type
        
    
    prediction = predictor.predict(results["silhouettes"])
    
    
        # unsure
        
    st.caption(f"Analysis performed using: **{model_type}**")

    st.subheader("Screening Results")

    with st.container(border=True):
        st.write(predictor.explain_prediction(prediction))

    st.subheader("Clinical Recommendation")

    with st.container(border=True):
        st.write(predictor.get_risk_assessment(prediction))

    st.caption(
        "⚠️ This tool is intended for preliminary screening only and does not "
        "provide a medical diagnosis. Please consult a qualified healthcare "
        "professional for clinical evaluation."
    )

    st.divider()
    st.caption("Created by Chanya Methaprayoon · Updated 2026")
    
    st.session_state.predicted_done = True


if __name__ == "__main__":
    main()