import streamlit as st
import cv2
import numpy as np
import os
import json
import warnings
import time
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
from fishai_model.inference import EmbeddingClassifier

# Page configuration
st.set_page_config(page_title="Fish Classifier", page_icon="🐟", layout="wide")


def read2rgb(img_path):
    target = cv2.imread(img_path)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    return target


@st.cache_resource
def load_model():
    """Load model with caching"""
    try:
        classifier = EmbeddingClassifier(
            os.path.join("fishai_model/model.ts"),
            os.path.join("fishai_model/database.pt"),
        )
        return classifier, True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, False


@st.cache_resource
def load_dict():
    """Load dictionary with caching"""
    try:
        with open("fish_dict.json", "r") as file:
            dict = json.load(file)
        return dict, True
    except Exception as e:
        st.error(f"Error loading dictionary: {str(e)}")
        return None, False

@st.cache_resource
def load_dishes():
    try:
        with open("fish_dishes.json", "r") as file:
            return json.load(file), True
    except Exception as e:
        st.warning("fish_dishes.json not found – using fallback")
        return {"dishes": {}, "fish_to_dishes": {}}, False

# Main app
def main():
    st.title("🐟 Fish Classification System")
    st.markdown("---")

    # Main content area
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.header("📷 Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a fish image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of fish for classification",
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = "temp_upload.jpg"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Display uploaded image
            image = cv2.imread(temp_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image_rgb, caption="Uploaded Image", width="stretch")

                # Image info
                st.markdown(f"""
                **Image Information:**
                - Size: {image.shape[1]} x {image.shape[0]} pixels
                - Format: {uploaded_file.type}
                """)

    with col2:
        st.header("🔬 Prediction Results")

        if uploaded_file is not None:

            model, model_exists = load_model()
            fish_dict, dict_exists = load_dict()

            if model is None:
                st.warning(f"""
                ⚠️ Model not found! .
                Please ensure the model files are in the correct location.
                """)
            else:
                # Make prediction
                with st.spinner("Analyzing image..."):
                    st.image("spinfish.gif", width="content")
                    time.sleep(1)  # Simulate processing time
                try:
                    # Extract features
                    features = read2rgb(temp_path)
                    if features is None:
                        st.error("Failed to read image")
                    else:
                        prediction = model.inference_numpy(features)

                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)

                        # Display results
                        confidence = prediction[0]["accuracy"] * 100
                        sysName = prediction[0]["name"]
                        commonName = fish_dict.get(sysName, "Unknown")

                        if confidence >= 50:
                            st.success(
                                f" Predicted Class: {commonName}\n\n" 
                                f" (System Name: {sysName})"
                                ,icon="✅"
                            )
                            st.metric("Confidence", f"{confidence:.2f}%")
                        else:
                            st.warning(
                                f" Low Confidence: {confidence:.2f}%\n\n"
                                f" Predicted: {commonName}\n\n"
                                f" (System Name: {sysName})"
                                ,icon="⚠️"
                            )

                        # Probability distribution
                        st.subheader("📊 Probability Distribution")
                        CLASS_NAMES = [res["name"] for res in prediction]
                        probabilities = np.array(
                            [res["accuracy"] for res in prediction]
                        )

                        # Create dataframe for display
                        prob_data = {
                            "Class": CLASS_NAMES,
                            "Probability (%)": [
                                f"{p*100:.2f}" for p in probabilities
                            ],
                            "Confidence": probabilities,
                        }

                        # Display as table with color coding
                        st.dataframe(
                            prob_data,
                            column_config={
                                "Class": "Fish Class",
                                "Probability (%)": "Probability",
                                "Confidence": st.column_config.ProgressColumn(
                                    "Confidence Score",
                                    format="%.4f",
                                    min_value=0,
                                    max_value=1,
                                ),
                            },
                            hide_index=True,
                            width="stretch",
                        )

                        # Probability bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = [
                            "#2ecc71" if i == prediction else "#95a5a6"
                            for i in range(len(CLASS_NAMES))
                        ]
                        bars = ax.bar(
                            CLASS_NAMES,
                            probabilities * 100,
                            color=colors,
                            edgecolor="black",
                            linewidth=0.5,
                        )

                        ax.set_ylabel("Probability (%)", fontsize=12)
                        ax.set_title(
                            "Classification Probabilities",
                            fontsize=14,
                            fontweight="bold",
                        )
                        ax.set_ylim(0, 100)
                        ax.axhline(
                            y=50,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label="Threshold (50%)",
                        )

                        # Add value labels on bars
                        for bar, prob in zip(bars, probabilities):
                            height = bar.get_height()
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                height,
                                f"{prob*100:.1f}%",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                                fontweight="bold",
                            )

                        plt.xticks(rotation=45, ha="right")
                        plt.legend()
                        plt.tight_layout()
                        plt.grid(axis="y", alpha=0.3)
                        st.pyplot(fig)

                        # Suggested Dishes Section
                        st.markdown("---")
                        st.subheader("🍽️ Suggested Dishes")
                        st.caption(f"Based on **{commonName}** ({sysName})")

                        dishes_data = load_dishes()[0]

                        dish_key = dishes_data.get("fish_to_dishes", {}).get(sysName)

                        if dish_key and dish_key in dishes_data.get("dishes", {}):
                            fish_dishes = dishes_data["dishes"][dish_key][:2]
                        else:
                            fish_dishes = [{"name": "This is not editable", "image_url": "https://picsum.photos/id/870/600/400"}]

                        if len(fish_dishes) == 1 and fish_dishes[0]["name"] == "This is not editable":
                            st.info("🍽️ This species is currently not configured for recipe suggestions.")
                            st.image(fish_dishes[0]["image_url"], caption="This is not editable", width="stretch")
                        else:
                            dcol1, dcol2 = st.columns(2)
                            for idx, dish in enumerate(fish_dishes):
                                with (dcol1 if idx == 0 else dcol2):
                                    st.image(dish["image_url"], caption=dish["name"], width="stretch")

                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    st.error(f"Error type: {type(e).__name__}")
                    st.info("""
                    **Troubleshooting:**
                    1. Ensure model was trained with same feature extraction
                    2. Check that class names match model classes
                    3. Verify model file is not corrupted
                    """)
        else:
            st.info("""
            👆 **Please upload an image to see predictions**
            
            **Tips for best results:**
            - Use clear, well-lit images
            - Ensure the fish is the main focus
            - Avoid blurry or dark images
            """)

            # Show example
            with st.expander("📋 Example Usage"):
                st.markdown("""
                1. **Upload a fish image** using the file uploader
                2. **View prediction results** with confidence scores
                """)

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style='text-align: center; color: #666;'>
        <p><strong>Fish Classification System</strong></p>
        <p>Traditional ML Classifiers</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
