"""Local Gradio prototype adapted from the original Colab notebook."""

from pathlib import Path

# ============================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================

import os
import shutil
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import gradio as gr

print("TensorFlow version:", tf.__version__)
print("Gradio version:", gr.__version__)
print("Libraries imported successfully.")

# ============================================================
# STEP 4: DEFINE SAVED MODEL PATHS
# ============================================================

PROJECT_ROOT = Path(os.environ.get("DEMENTIA_PROJECT_ROOT", "ADNI_Project"))
MODEL_FOLDER = PROJECT_ROOT / "Interface_Models"

MRI_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "best_mri_3slice_mobilenet_model.keras"
)

COGNITIVE_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "cognitive_random_forest.joblib"
)

COGNITIVE_FEATURES_PATH = os.path.join(
    MODEL_FOLDER,
    "cognitive_feature_columns.joblib"
)

FUSION_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "fusion_random_forest.joblib"
)

FUSION_FEATURES_PATH = os.path.join(
    MODEL_FOLDER,
    "fusion_feature_columns.joblib"
)

required_files = {
    "MRI model": MRI_MODEL_PATH,
    "Cognitive model": COGNITIVE_MODEL_PATH,
    "Cognitive feature list": COGNITIVE_FEATURES_PATH,
    "Fusion model": FUSION_MODEL_PATH,
    "Fusion feature list": FUSION_FEATURES_PATH
}

print("Checking saved files:\n")

all_files_found = True

for name, path in required_files.items():
    if os.path.isfile(path):
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"✓ {name}: {size_mb:.2f} MB")
    else:
        print(f"✗ Missing: {name}")
        print(f"  Expected path: {path}")
        all_files_found = False

if not all_files_found:
    raise FileNotFoundError(
        "One or more saved model files are missing. "
        "Please check the paths above."
    )

print("\nAll saved model files were found.")

# ============================================================
# STEP 5: LOAD SAVED MODELS
# ============================================================

print("Loading MRI model...")
mri_model = tf.keras.models.load_model(MRI_MODEL_PATH)

print("Loading cognitive model...")
cognitive_model = joblib.load(COGNITIVE_MODEL_PATH)

print("Loading cognitive feature order...")
cognitive_feature_columns = joblib.load(
    COGNITIVE_FEATURES_PATH
)

print("Loading fusion model...")
fusion_model = joblib.load(FUSION_MODEL_PATH)

print("Loading fusion feature order...")
fusion_feature_columns = joblib.load(
    FUSION_FEATURES_PATH
)

print("\nAll models loaded successfully.")

print("\nMRI input shape:", mri_model.input_shape)
print("MRI output shape:", mri_model.output_shape)

print("\nCognitive features:")
print(cognitive_feature_columns)

print("\nFusion features:")
print(fusion_feature_columns)

# ============================================================
# STEP 7: LOAD MATCHED COGNITIVE DATA
# ============================================================

COGNITIVE_DATA_PATH = (
    PROJECT_ROOT / "Cognitive_Data/final_cognitive_702.csv"
)

final_cognitive = pd.read_csv(COGNITIVE_DATA_PATH)

print("Cognitive dataset loaded successfully.")
print("Shape:", final_cognitive.shape)
print("Unique participants:", final_cognitive["RID"].nunique())
print("\nColumns:")
print(final_cognitive.columns.tolist())

# ============================================================
# STEP 8: LOCATE PROCESSED MRI FILES
# ============================================================

import re

MRI_FOLDER = (
    PROJECT_ROOT / "MRI_Images/Processed_MRI_160"
)

mri_files = sorted(
    file for file in os.listdir(MRI_FOLDER)
    if file.endswith(".npy")
)

mri_records = []

for file in mri_files:
    match = re.search(r"RID_(.*?)_IMG", file)

    if match:
        rid = int(match.group(1))

        mri_records.append({
            "RID": rid,
            "MRI_FILE": file,
            "MRI_PATH": os.path.join(MRI_FOLDER, file)
        })

mri_lookup = pd.DataFrame(mri_records)

print("MRI lookup created successfully.")
print("MRI files found:", len(mri_lookup))
print("Unique RIDs:", mri_lookup["RID"].nunique())
print("\nFirst five records:")

# ============================================================
# STEP 9: VERIFY COGNITIVE AND MRI MATCHING
# ============================================================

sample_data = final_cognitive.merge(
    mri_lookup,
    on="RID",
    how="inner"
)

print("Matched sample-patient records:", sample_data.shape)
print("Unique matched participants:", sample_data["RID"].nunique())

if sample_data["RID"].nunique() != 702:
    print(
        "Warning: The number of matched participants is not 702."
    )
else:
    print("All 702 participants matched successfully.")

# ============================================================
# STEP 10: MRI LOADING AND PREDICTION FUNCTIONS
# ============================================================

import numpy as np

IMG_SIZE = 160
MRI_SLICE_INDICES = [60, 80, 100]
CLASS_NAMES = ["CN", "MCI", "AD"]


def prepare_mri_input(mri_path):
    """
    Load one preprocessed 160x160x160 MRI volume and extract
    slices 60, 80 and 100 as the three input channels.
    """

    if not isinstance(mri_path, str) or not mri_path:
        raise ValueError("A valid MRI file path was not provided.")

    if not os.path.isfile(mri_path):
        raise FileNotFoundError(f"MRI file not found: {mri_path}")

    if not mri_path.lower().endswith(".npy"):
        raise ValueError(
            "The MRI file must be a preprocessed .npy volume."
        )

    volume = np.load(mri_path).astype(np.float32)

    if volume.shape != (160, 160, 160):
        raise ValueError(
            "Unexpected MRI dimensions. "
            f"Expected (160, 160, 160), but received {volume.shape}."
        )

    # Do not normalise again:
    # these stored volumes were already normalised during preprocessing.
    three_slice_image = np.stack(
        [
            volume[:, :, MRI_SLICE_INDICES[0]],
            volume[:, :, MRI_SLICE_INDICES[1]],
            volume[:, :, MRI_SLICE_INDICES[2]]
        ],
        axis=-1
    )

    # Add the batch dimension:
    # (160, 160, 3) -> (1, 160, 160, 3)
    model_input = np.expand_dims(
        three_slice_image,
        axis=0
    ).astype(np.float32)

    return volume, three_slice_image, model_input


def predict_mri(mri_path):
    """
    Generate CN, MCI and AD probabilities from one MRI volume.
    """

    volume, three_slice_image, model_input = prepare_mri_input(
        mri_path
    )

    probabilities = mri_model.predict(
        model_input,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    result = {
        "predicted_index": predicted_index,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities
            )
        },
        "volume": volume,
        "three_slice_image": three_slice_image
    }

    return result

# ============================================================
# STEP 11: COGNITIVE MODEL PREDICTION FUNCTION
# ============================================================

def predict_cognitive(cognitive_values):
    """
    cognitive_values must be a dictionary containing the
    16 cognitive feature names and their numerical values.
    """

    missing_features = [
        feature
        for feature in cognitive_feature_columns
        if feature not in cognitive_values
    ]

    if missing_features:
        raise ValueError(
            "Missing cognitive features: "
            + ", ".join(missing_features)
        )

    cognitive_input = pd.DataFrame(
        [[cognitive_values[feature]
          for feature in cognitive_feature_columns]],
        columns=cognitive_feature_columns
    )

    predicted_label = int(
        cognitive_model.predict(cognitive_input)[0]
    )

    probabilities = cognitive_model.predict_proba(
        cognitive_input
    )[0]

    # Cognitive model labels are 1=CN, 2=MCI and 3=AD
    predicted_index = predicted_label - 1

    result = {
        "predicted_label": predicted_label,
        "predicted_index": predicted_index,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities
            )
        },
        "input_dataframe": cognitive_input
    }

    return result

# ============================================================
# STEP 12: MULTIMODAL FUSION PREDICTION FUNCTION
# ============================================================

def predict_fusion(cognitive_values, mri_probabilities):
    """
    Combine 16 cognitive variables with three MRI probability
    features and run the saved fusion Random Forest model.
    """

    fusion_values = dict(cognitive_values)

    fusion_values.update({
        "MRI_PROB_CN": mri_probabilities["CN"],
        "MRI_PROB_MCI": mri_probabilities["MCI"],
        "MRI_PROB_AD": mri_probabilities["AD"]
    })

    missing_features = [
        feature
        for feature in fusion_feature_columns
        if feature not in fusion_values
    ]

    if missing_features:
        raise ValueError(
            "Missing fusion features: "
            + ", ".join(missing_features)
        )

    fusion_input = pd.DataFrame(
        [[fusion_values[feature]
          for feature in fusion_feature_columns]],
        columns=fusion_feature_columns
    )

    predicted_index = int(
        fusion_model.predict(fusion_input)[0]
    )

    probabilities = fusion_model.predict_proba(
        fusion_input
    )[0]

    result = {
        "predicted_index": predicted_index,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index]),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(
                CLASS_NAMES,
                probabilities
            )
        },
        "input_dataframe": fusion_input
    }

    return result

# ============================================================
# STEP 13: COMPLETE SAMPLE-PATIENT PREDICTION ENGINE
# ============================================================

def predict_sample_patient(rid):
    """
    Run MRI-only, cognitive-only and multimodal fusion
    predictions for one matched ADNI participant.
    """

    try:
        rid = int(rid)
    except (TypeError, ValueError):
        raise ValueError("RID must be a valid integer.")

    patient_rows = sample_data[
        sample_data["RID"] == rid
    ]

    if patient_rows.empty:
        raise ValueError(
            f"RID {rid} was not found in the matched dataset."
        )

    patient = patient_rows.iloc[0]

    cognitive_values = {
        feature: float(patient[feature])
        for feature in cognitive_feature_columns
    }

    mri_result = predict_mri(
        patient["MRI_PATH"]
    )

    cognitive_result = predict_cognitive(
        cognitive_values
    )

    fusion_result = predict_fusion(
        cognitive_values,
        mri_result["probabilities"]
    )

    actual_label = int(patient["DIAGNOSIS"])
    actual_class = CLASS_NAMES[actual_label - 1]

    return {
        "RID": rid,
        "actual_class": actual_class,
        "MRI": mri_result,
        "Cognitive": cognitive_result,
        "Fusion": fusion_result
    }

# ============================================================
# STEP 15: INTERFACE HELPER FUNCTIONS
# ============================================================

import matplotlib.pyplot as plt


def format_model_result(model_name, result):
    """Create a polished HTML result card for one model."""

    prediction = result["predicted_class"]
    confidence = result["confidence"] * 100

    model_styles = {
        "MRI-only model": ("#246B9E", "#EAF4FB", "MRI"),
        "Cognitive-only model": ("#7451A6", "#F3EEFA", "Cognitive"),
        "Multimodal fusion model": ("#087E8B", "#EAF8F7", "Fusion")
    }
    accent, tint, short_name = model_styles.get(
        model_name, ("#087E8B", "#EAF8F7", "Model")
    )

    return f"""
    <article style="
        border: 1px solid #d9e5ec;
        border-top: 5px solid {accent};
        border-radius: 16px;
        padding: 24px;
        background: linear-gradient(145deg, #ffffff 60%, {tint});
        min-height: 180px;
        box-shadow: 0 8px 24px rgba(28,60,80,0.08);
    ">
        <div style="
            display:inline-block; padding:5px 9px; border-radius:999px;
            background:{tint}; color:{accent}; font-size:12px;
            font-weight:750; letter-spacing:.04em; text-transform:uppercase;
        ">{short_name} assessment</div>
        <h3 style="margin:14px 0 18px; color:#163a5f; font-size:19px;">
            {model_name}
        </h3>
        <p style="margin:0 0 5px; color:#62778a; font-size:13px; font-weight:650;">
            Predicted class
        </p>
        <div style="display:flex; align-items:flex-end; justify-content:space-between; gap:16px;">
            <strong style="font-size:30px; line-height:1; color:{accent};">{prediction}</strong>
            <span style="color:#52667a; font-size:14px;">
                Confidence <strong style="color:#163a5f;">{confidence:.2f}%</strong>
            </span>
        </div>
    </article>
    """


def create_probability_chart(result, title):
    """Create a spacious, colour-coded probability chart."""

    probabilities = result["probabilities"]
    labels = list(probabilities.keys())
    values = [probability * 100 for probability in probabilities.values()]

    if "MRI" in title:
        palette = ["#9CC8E3", "#4E95C2", "#246B9E"]
        accent = "#246B9E"
        display_title = "MRI probability distribution"
    elif "Cognitive" in title:
        palette = ["#C8B5E3", "#9A78C2", "#7451A6"]
        accent = "#7451A6"
        display_title = "Cognitive probability distribution"
    else:
        palette = ["#9AD9D2", "#43AAA8", "#087E8B"]
        accent = "#087E8B"
        display_title = "Fusion probability distribution"

    fig, ax = plt.subplots(figsize=(5.4, 4.2), facecolor="white")
    bars = ax.bar(labels, values, color=palette, width=0.62)

    ax.set_ylim(0, 112)
    ax.set_ylabel("Probability (%)", color="#52667A", labelpad=10)
    ax.set_title(
        display_title,
        color="#163A5F",
        fontsize=12,
        fontweight="bold",
        pad=18
    )
    ax.grid(axis="y", color="#D9E5EC", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8CBD6")
    ax.spines["bottom"].set_color("#B8CBD6")
    ax.tick_params(colors="#52667A")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 3, 105),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            color=accent,
            fontweight="bold",
            fontsize=10
        )

    fig.subplots_adjust(left=0.18, right=0.96, top=0.82, bottom=0.16)
    return fig


def create_mri_preview(three_slice_image):
    """Present the three MRI slices in a polished clinical-style panel."""

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), facecolor="#F7FAFC")

    slice_titles = ["Slice 60", "Slice 80", "Slice 100"]

    for index, axis in enumerate(axes):
        axis.imshow(
            three_slice_image[:, :, index],
            cmap="gray",
            interpolation="bilinear"
        )
        axis.set_title(
            slice_titles[index],
            fontsize=11,
            fontweight="bold",
            color="#29475B",
            pad=12
        )
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("#071A2A")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("#2CA6A4")
            spine.set_linewidth(1.4)

    fig.suptitle(
        "MRI Slices",
        fontsize=17,
        fontweight="bold",
        color="#123B5D",
        y=0.97
    )
    fig.text(
        0.5,
        0.885,
        "Axial views used by the MRI assessment model",
        ha="center",
        color="#62778A",
        fontsize=10
    )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.78, bottom=0.08, wspace=0.12)
    return fig

# ============================================================
# STEP 16: SAMPLE PATIENT INTERFACE FUNCTION
# ============================================================

def run_sample_patient_prediction(selected_rid):
    """
    Run MRI-only, cognitive-only and fusion predictions
    for one selected participant and prepare all Gradio outputs.
    """

    try:
        selected_rid = int(selected_rid)

        # Step 13 already defines this complete prediction engine.
        result = predict_sample_patient(selected_rid)

        actual_class = result["actual_class"]
        mri_result = result["MRI"]
        cognitive_result = result["Cognitive"]
        fusion_result = result["Fusion"]

        # ----------------------------------------------------
        # Actual diagnosis information
        # ----------------------------------------------------
        actual_html = f"""
        <div style="
            padding: 15px;
            border-radius: 10px;
            background: #eef5fa;
            border-left: 5px solid #315b7d;
        ">
            <strong>Participant RID:</strong> {selected_rid}
            <br>
            <strong>Recorded diagnosis:</strong> {actual_class}
            <br>
            <strong>Assessment type:</strong> Sample ADNI participant
        </div>
        """

        # ----------------------------------------------------
        # Prediction result cards
        # ----------------------------------------------------
        mri_html = format_model_result(
            "MRI-only model",
            mri_result
        )

        cognitive_html = format_model_result(
            "Cognitive-only model",
            cognitive_result
        )

        fusion_html = format_model_result(
            "Multimodal fusion model",
            fusion_result
        )

        # ----------------------------------------------------
        # Probability charts
        # ----------------------------------------------------
        mri_prob_plot = create_probability_chart(
            mri_result,
            "MRI Model Probabilities"
        )

        cognitive_prob_plot = create_probability_chart(
            cognitive_result,
            "Cognitive Model Probabilities"
        )

        fusion_prob_plot = create_probability_chart(
            fusion_result,
            "Fusion Model Probabilities"
        )

        # ----------------------------------------------------
        # MRI slice preview
        # ----------------------------------------------------
        mri_preview_plot = create_mri_preview(
            mri_result["three_slice_image"]
        )

        return (
            actual_html,
            mri_html,
            cognitive_html,
            fusion_html,
            mri_prob_plot,
            cognitive_prob_plot,
            fusion_prob_plot,
            mri_preview_plot
        )

    except Exception as e:

        error_html = f"""
        <div style="
            padding: 15px;
            border-radius: 10px;
            background: #ffe6e6;
            color: #990000;
            border: 1px solid #cc0000;
        ">
            <h3>Prediction Error</h3>
            <p>{str(e)}</p>
        </div>
        """

        return (
            error_html,
            error_html,
            error_html,
            error_html,
            None,
            None,
            None,
            None
        )


print("Step 16 completed successfully.")
print(
    "Sample prediction function available:",
    callable(run_sample_patient_prediction)
)

# ============================================================
# STEP 19: EXTRACT VALID INPUT RANGES
# ============================================================

print("=" * 80)
print("VALID INPUT RANGES FROM TRAINING DATA")
print("=" * 80)

feature_ranges = {}

for feature in cognitive_feature_columns:

    minimum = final_cognitive[feature].min()
    maximum = final_cognitive[feature].max()

    feature_ranges[feature] = {
        "min": minimum,
        "max": maximum
    }

    print(f"{feature:<20} {minimum:>8}   →   {maximum}")

# ============================================================
# STEP 21: MANUAL-ENTRY PREDICTION FUNCTION
# ============================================================

GENDER_MAP = {
    "Male": 0,
    "Female": 1
}

# Valid values present in the matched training cohort
CDR_MEMORY_VALUES = sorted(
    final_cognitive["CDMEMORY"].dropna().unique().tolist()
)

CDR_ORIENTATION_VALUES = sorted(
    final_cognitive["CDORIENT"].dropna().unique().tolist()
)

CDR_JUDGEMENT_VALUES = sorted(
    final_cognitive["CDJUDGE"].dropna().unique().tolist()
)

CDR_COMMUNITY_VALUES = sorted(
    final_cognitive["CDCOMMUN"].dropna().unique().tolist()
)

CDR_HOME_VALUES = sorted(
    final_cognitive["CDHOME"].dropna().unique().tolist()
)

CDR_CARE_VALUES = sorted(
    final_cognitive["CDCARE"].dropna().unique().tolist()
)

CDR_GLOBAL_VALUES = sorted(
    final_cognitive["CDGLOBAL"].dropna().unique().tolist()
)


def validate_range(name, value, minimum, maximum):
    """
    Validate that a manual numerical input is present and lies
    inside the permitted range.
    """

    if value is None:
        raise gr.Error(f"Please enter a value for {name}.")

    value = float(value)

    if value < minimum or value > maximum:
        raise gr.Error(
            f"{name} must be between {minimum} and {maximum}."
        )

    return value


def resolve_manual_mri_path(
    mri_source,
    sample_mri_rid,
    uploaded_mri
):
    """
    Select either an existing sample MRI or an uploaded .npy MRI.
    """

    if mri_source == "Use sample MRI":

        if sample_mri_rid is None:
            raise gr.Error("Please select a sample MRI participant.")

        matching_rows = mri_lookup[
            mri_lookup["RID"] == int(sample_mri_rid)
        ]

        if matching_rows.empty:
            raise gr.Error(
                f"No MRI file was found for RID {sample_mri_rid}."
            )

        return matching_rows.iloc[0]["MRI_PATH"], (
            f"Sample MRI from RID {int(sample_mri_rid)}"
        )

    if mri_source == "Upload processed MRI":

        if uploaded_mri is None:
            raise gr.Error(
                "Please upload a processed MRI .npy file."
            )

        uploaded_path = str(uploaded_mri)

        if not uploaded_path.lower().endswith(".npy"):
            raise gr.Error(
                "The uploaded MRI must be a .npy file."
            )

        return uploaded_path, "Uploaded MRI volume"

    raise gr.Error("Please select an MRI source.")


def run_manual_prediction(
    age,
    gender,
    education,
    mmse,
    delayed_recall,
    cdr_memory,
    cdr_orientation,
    cdr_judgement,
    cdr_community,
    cdr_home,
    cdr_care,
    cdr_global,
    cdr_sum_boxes,
    adas_total,
    adas_13,
    faq_total,
    mri_source,
    sample_mri_rid,
    uploaded_mri
):
    """
    Run MRI-only, cognitive-only and multimodal predictions
    using manually entered cognitive values.
    """

    # --------------------------------------------------------
    # Validate manual values
    # --------------------------------------------------------

    age = validate_range("Age", age, 55, 91)
    education = validate_range(
        "Years of education",
        education,
        4,
        20
    )

    mmse = validate_range(
        "MMSE score",
        mmse,
        20,
        30
    )

    delayed_recall = validate_range(
        "Delayed recall",
        delayed_recall,
        0,
        3
    )

    cdr_sum_boxes = validate_range(
        "CDR Sum of Boxes",
        cdr_sum_boxes,
        0,
        9
    )

    adas_total = validate_range(
        "ADAS total score",
        adas_total,
        0,
        42.67
    )

    adas_13 = validate_range(
        "ADAS-Cog 13 score",
        adas_13,
        1,
        54.67
    )

    # -1 was present in the dataset but is not accepted as a
    # meaningful manual score.
    faq_total = validate_range(
        "FAQ total score",
        faq_total,
        0,
        27
    )

    if gender not in GENDER_MAP:
        raise gr.Error("Please select Male or Female.")

    # --------------------------------------------------------
    # Construct the exact cognitive feature dictionary
    # --------------------------------------------------------

    cognitive_values = {
        "MMSCORE": mmse,
        "DELAYED_RECALL": delayed_recall,
        "CDMEMORY": float(cdr_memory),
        "CDORIENT": float(cdr_orientation),
        "CDJUDGE": float(cdr_judgement),
        "CDCOMMUN": float(cdr_community),
        "CDHOME": float(cdr_home),
        "CDCARE": float(cdr_care),
        "CDGLOBAL": float(cdr_global),
        "CDRSB": cdr_sum_boxes,
        "TOTSCORE": adas_total,
        "TOTAL13": adas_13,
        "FAQTOTAL": faq_total,
        "GENDER": GENDER_MAP[gender],
        "PTEDUCAT": education,
        "AGE": age
    }

    # --------------------------------------------------------
    # Resolve and validate the MRI
    # --------------------------------------------------------

    mri_path, mri_description = resolve_manual_mri_path(
        mri_source,
        sample_mri_rid,
        uploaded_mri
    )

    # The existing MRI function checks:
    # extension, dimensions and successful loading.
    mri_result = predict_mri(mri_path)

    cognitive_result = predict_cognitive(
        cognitive_values
    )

    fusion_result = predict_fusion(
        cognitive_values,
        mri_result["probabilities"]
    )

    input_summary = f"""
    <div style="
        padding: 14px;
        border-radius: 10px;
        background: #eef5fa;
        border-left: 5px solid #315b7d;
    ">
        <strong>Assessment type:</strong> Manual entry
        <br>
        <strong>MRI source:</strong> {mri_description}
        <br>
        <strong>MRI dimensions:</strong> 160 × 160 × 160
        <br>
        <strong>Gender:</strong> {gender}
        <br>
        <strong>Age:</strong> {age:.0f} years
    </div>
    """

    mri_card = format_model_result(
        "MRI-only model",
        mri_result
    )

    cognitive_card = format_model_result(
        "Cognitive-only model",
        cognitive_result
    )

    fusion_card = format_model_result(
        "Multimodal fusion model",
        fusion_result
    )

    mri_probability_chart = create_probability_chart(
        mri_result,
        "MRI Model Probabilities"
    )

    cognitive_probability_chart = create_probability_chart(
        cognitive_result,
        "Cognitive Model Probabilities"
    )

    fusion_probability_chart = create_probability_chart(
        fusion_result,
        "Fusion Model Probabilities"
    )

    mri_preview = create_mri_preview(
        mri_result["three_slice_image"]
    )

    return (
        input_summary,
        mri_card,
        cognitive_card,
        fusion_card,
        mri_probability_chart,
        cognitive_probability_chart,
        fusion_probability_chart,
        mri_preview
    )

# ============================================================
# STEP 17: BUILD GRADIO INTERFACE
# ============================================================

# ============================================================
# STEP 17A: CREATE MAIN INTERFACE AND HOME TAB
# ============================================================

# Create participant lists for dropdowns
participant_rids = sorted(
    sample_data["RID"]
    .astype(int)
    .unique()
    .tolist()
)

participant_choices = [
    (f"Participant RID {rid}", rid)
    for rid in participant_rids
]

print("Number of available participants:", len(participant_rids))


custom_css = """
:root {
    --ink: #102A43;
    --muted: #52667A;
    --navy: #123B5D;
    --teal: #087E8B;
    --teal-dark: #05616B;
    --aqua: #EAF8F7;
    --line: #D9E5EC;
    --surface: #FFFFFF;
    --canvas: #F4F8FA;
    --warning: #FFF8E8;
}

body, .gradio-container {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif !important;
    color: var(--ink) !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding: 0 24px 44px !important;
    background: var(--canvas) !important;
}

.app-hero {
    position: relative;
    overflow: hidden;
    margin: 0 -24px 24px;
    padding: 44px 48px 40px;
    color: white;
    background:
        radial-gradient(circle at 88% 10%, rgba(71, 209, 196, .28), transparent 30%),
        linear-gradient(135deg, #102F4A 0%, #135270 58%, #087E8B 100%);
    border-radius: 0 0 28px 28px;
    box-shadow: 0 14px 40px rgba(16, 47, 74, .18);
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 18px;
    padding: 7px 12px;
    border: 1px solid rgba(255,255,255,.28);
    border-radius: 999px;
    background: rgba(255,255,255,.12);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}

.hero-badge::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #63E6D6;
    box-shadow: 0 0 0 4px rgba(99,230,214,.14);
}

.app-hero h1 {
    max-width: 840px;
    margin: 0 0 12px;
    color: #FFFFFF !important;
    font-size: clamp(32px, 5vw, 52px) !important;
    line-height: 1.08 !important;
    letter-spacing: -.035em;
}

.app-hero p {
    max-width: 740px;
    margin: 0;
    color: rgba(255,255,255,.82) !important;
    font-size: 17px;
    line-height: 1.65;
}

.hero-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 24px;
}

.hero-meta span {
    padding: 7px 11px;
    border-radius: 10px;
    background: rgba(255,255,255,.11);
    color: rgba(255,255,255,.9);
    font-size: 13px;
}

.tabs {
    border: 0 !important;
}

.tab-nav {
    gap: 6px !important;
    margin-bottom: 20px !important;
    padding: 6px !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    background: var(--surface) !important;
    box-shadow: 0 4px 18px rgba(28, 60, 80, .06);
}

.tab-nav button {
    min-height: 42px !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-weight: 650 !important;
}

.tab-nav button.selected {
    color: var(--navy) !important;
    background: var(--aqua) !important;
    border-color: transparent !important;
}

.gr-group, .gr-box, .block, .form {
    border-color: var(--line) !important;
}

.block:not(.plot-container) {
    border-radius: 14px !important;
}

.form {
    background: var(--surface) !important;
    border-radius: 14px !important;
}

h2, h3 {
    color: var(--navy) !important;
    letter-spacing: -.015em;
}

h2 {
    margin-top: 24px !important;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--line);
}

h3 {
    margin-top: 20px !important;
}

label span {
    color: #334E68 !important;
    font-weight: 650 !important;
}

input, textarea, select {
    border-radius: 10px !important;
}

button.primary {
    min-height: 46px !important;
    border: 0 !important;
    border-radius: 11px !important;
    color: white !important;
    background: linear-gradient(135deg, var(--teal), var(--teal-dark)) !important;
    box-shadow: 0 8px 18px rgba(8, 126, 139, .22) !important;
    font-weight: 750 !important;
    transition: transform .16s ease, box-shadow .16s ease !important;
}

button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 11px 24px rgba(8, 126, 139, .28) !important;
}

.prose blockquote {
    margin: 18px 0 !important;
    padding: 14px 16px !important;
    border-left: 4px solid #D99B28 !important;
    border-radius: 0 10px 10px 0;
    background: var(--warning) !important;
    color: #6C4A12 !important;
}

.plot-container, .gradio-container canvas {
    border-radius: 14px !important;
}


/* High-contrast tabs and actions */
.tab-nav button,
.tab-nav button[role="tab"] {
    color: #17324D !important;
    background: #E8F1F5 !important;
    border: 1px solid #C9DAE3 !important;
    opacity: 1 !important;
}

.tab-nav button:hover,
.tab-nav button[role="tab"]:hover {
    color: #FFFFFF !important;
    background: #176B78 !important;
}

.tab-nav button.selected,
.tab-nav button[aria-selected="true"] {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #123B5D, #087E8B) !important;
    border-color: transparent !important;
    box-shadow: 0 5px 14px rgba(18, 59, 93, .20) !important;
}

button.primary, button.primary *,
.primary, .primary * {
    color: #FFFFFF !important;
}

.result-card {
    min-width: 300px !important;
    min-height: 205px !important;
}

.result-card > div {
    height: 100%;
}

.probability-chart {
    min-width: 320px !important;
    min-height: 420px !important;
    padding: 12px !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    background: #FFFFFF !important;
    box-shadow: 0 6px 20px rgba(28, 60, 80, .07);
}

.mri-preview-chart {
    min-height: 430px !important;
    padding: 14px !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    background: #FFFFFF !important;
}


/* Explicit readability overrides for Gradio light/dark theme variables */
.gradio-container {
    --body-text-color: #183247 !important;
    --body-text-color-subdued: #52667A !important;
    --block-label-text-color: #29475B !important;
    --input-label-text-color: #29475B !important;
    --button-secondary-text-color: #5B2D0A !important;
    --button-secondary-text-color-hover: #FFFFFF !important;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose em,
.gradio-container .markdown,
.gradio-container .markdown p,
.gradio-container .markdown li,
.gradio-container .markdown strong,
.gradio-container .markdown em {
    color: #183247 !important;
    opacity: 1 !important;
}

.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .markdown h1,
.gradio-container .markdown h2,
.gradio-container .markdown h3 {
    color: #123B5D !important;
    opacity: 1 !important;
}

.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap,
.gradio-container .label-wrap span {
    color: #29475B !important;
    opacity: 1 !important;
}

/* Warm orange navigation with permanently readable labels */
.gradio-container button[role="tab"],
.gradio-container .tab-nav button {
    color: #63310B !important;
    -webkit-text-fill-color: #63310B !important;
    background: #FFF0E1 !important;
    border: 1px solid #F0B47B !important;
    opacity: 1 !important;
    font-weight: 800 !important;
    text-shadow: none !important;
}

.gradio-container button[role="tab"]:hover,
.gradio-container .tab-nav button:hover {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background: #E8871E !important;
    border-color: #E8871E !important;
}

.gradio-container button[role="tab"][aria-selected="true"],
.gradio-container .tab-nav button.selected {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background: linear-gradient(135deg, #E8871E, #C9670A) !important;
    border-color: #C9670A !important;
    box-shadow: 0 6px 16px rgba(201, 103, 10, .24) !important;
}

/* Keep the hero intentionally light on its dark background */
.gradio-container .app-hero,
.gradio-container .app-hero h1,
.gradio-container .app-hero p,
.gradio-container .app-hero span,
.gradio-container .app-hero div {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}


/* White values inside blue input controls */
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap input,
.gradio-container .secondary-wrap input,
.gradio-container [role="combobox"],
.gradio-container [role="combobox"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}


/* Readable values inside opened dropdown menus */
.gradio-container [role="listbox"] [role="option"],
.gradio-container [role="listbox"] [role="option"] *,
.gradio-container .options li,
.gradio-container .options li *,
.gradio-container ul.options li,
.gradio-container ul.options li *,
.gradio-container .dropdown-option,
.gradio-container .dropdown-option * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
}

.gradio-container [role="listbox"] [role="option"]:hover,
.gradio-container .options li:hover,
.gradio-container .dropdown-option:hover {
    background: #173B55 !important;
}

.gradio-container [role="listbox"] [role="option"][aria-selected="true"],
.gradio-container .options li.selected {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    background: #123B5D !important;
}

footer { display: none !important; }

@media (max-width: 700px) {
    .gradio-container { padding: 0 14px 28px !important; }
    .app-hero { margin: 0 -14px 18px; padding: 32px 22px 28px; border-radius: 0 0 22px 22px; }
    .app-hero h1 { font-size: 32px !important; }
    .app-hero p { font-size: 15px; }
}
"""


dementia_interface = gr.Blocks(
    title="AI-Based Multimodal Dementia Early Detection Tool",
    css=custom_css
)


with dementia_interface:

    gr.HTML(
        """
        <header class="app-hero">
            <div class="hero-badge">Academic research prototype</div>
            <h1>AI Multimodal Dementia Early Detection</h1>
            <p>
                A clear, research-focused workspace combining structural MRI,
                cognitive biomarkers and multimodal fusion for three-class
                dementia risk assessment.
            </p>
            <div class="hero-meta" aria-label="Assessment modalities">
                <span>Structural MRI</span>
                <span>Cognitive assessment</span>
                <span>Multimodal fusion</span>
            </div>
        </header>
        """
    )

    interface_tabs = gr.Tabs()

    with interface_tabs:

        with gr.Tab("Home"):

            gr.Markdown(
                """
                ## Research Prototype

                This prototype demonstrates an artificial
                intelligence framework that compares:

                - an MRI-only deep learning model;
                - a cognitive-only machine learning model;
                - a multimodal fusion model.

                The system classifies participants into:

                - **CN:** Cognitively Normal
                - **MCI:** Mild Cognitive Impairment
                - **AD:** Alzheimer's Disease

                > **Important:** This application was developed
                > for academic research and demonstration only.
                > It must not be used for clinical diagnosis or
                > treatment decisions.
                """
            )

print("Step 17A completed.")

# ============================================================
# STEP 17B: CREATE SAMPLE PATIENT TAB
# ============================================================

with dementia_interface:

    with interface_tabs:

        with gr.Tab("Sample Patient"):

            gr.Markdown(
                """
                ## Sample Participant Assessment

                Select one matched ADNI participant. The system
                will automatically load the participant's
                processed MRI scan and cognitive assessment data.
                """
            )

            with gr.Row():

                rid_dropdown = gr.Dropdown(
                    choices=participant_choices,
                    value=participant_rids[0],
                    label=(
                        f"Select participant "
                        f"({len(participant_rids)} matched participants)"
                    )
                )

                sample_predict_button = gr.Button(
                    "Run Assessment",
                    variant="primary"
                )

            actual_diagnosis_output = gr.HTML()

            gr.Markdown("## Model Predictions")

            with gr.Row():

                mri_result_output = gr.HTML(elem_classes=["result-card"])

                cognitive_result_output = gr.HTML(elem_classes=["result-card"])

                fusion_result_output = gr.HTML(elem_classes=["result-card"])

            gr.Markdown("## Model Probability Distributions")

            with gr.Row():

                mri_probability_output = gr.Plot(elem_classes=["probability-chart"])

                cognitive_probability_output = gr.Plot(elem_classes=["probability-chart"])

                fusion_probability_output = gr.Plot(elem_classes=["probability-chart"])

            gr.Markdown("## MRI Input Preview")

            mri_preview_output = gr.Plot(elem_classes=["mri-preview-chart"])

            sample_predict_button.click(
                fn=run_sample_patient_prediction,
                inputs=[rid_dropdown],
                outputs=[
                    actual_diagnosis_output,
                    mri_result_output,
                    cognitive_result_output,
                    fusion_result_output,
                    mri_probability_output,
                    cognitive_probability_output,
                    fusion_probability_output,
                    mri_preview_output
                ]
            )

print("Step 17B completed.")

# ============================================================
# STEP 17C: CREATE MANUAL ENTRY TAB
# ============================================================

with dementia_interface:

    with interface_tabs:

        with gr.Tab("Manual Entry"):

            gr.Markdown(
                """
                ## Manual Dementia Risk Assessment

                Enter cognitive and demographic values and select
                a processed MRI source.

                All displayed ranges were derived from the final
                matched cohort used for model development.

                > The model confidence values are not clinical
                > probabilities and must not be interpreted as a
                > confirmed medical diagnosis.
                """
            )

            # ------------------------------------------------
            # PARTICIPANT INFORMATION
            # ------------------------------------------------

            gr.Markdown("### 1. Participant Information")

            with gr.Row():

                manual_age = gr.Number(
                    label="Age (55–91 years)",
                    value=70,
                    minimum=55,
                    maximum=91,
                    step=1
                )

                manual_gender = gr.Dropdown(
                    choices=["Male", "Female"],
                    value="Male",
                    label="Gender"
                )

                manual_education = gr.Number(
                    label="Years of education (4–20)",
                    value=12,
                    minimum=4,
                    maximum=20,
                    step=1
                )

            # ------------------------------------------------
            # COGNITIVE ASSESSMENT
            # ------------------------------------------------

            gr.Markdown("### 2. Cognitive Assessment Scores")

            with gr.Row():

                manual_mmse = gr.Number(
                    label="MMSE score (20–30)",
                    value=27,
                    minimum=20,
                    maximum=30,
                    step=1
                )

                manual_delayed_recall = gr.Number(
                    label="Delayed recall (0–3)",
                    value=2,
                    minimum=0,
                    maximum=3,
                    step=1
                )

                manual_faq = gr.Number(
                    label="FAQ total score (0–27)",
                    value=3,
                    minimum=0,
                    maximum=27,
                    step=1
                )

            with gr.Row():

                manual_adas_total = gr.Number(
                    label="ADAS total score (0–42.67)",
                    value=10,
                    minimum=0,
                    maximum=42.67,
                    step=0.01
                )

                manual_adas_13 = gr.Number(
                    label="ADAS-Cog 13 score (1–54.67)",
                    value=15,
                    minimum=1,
                    maximum=54.67,
                    step=0.01
                )

            # ------------------------------------------------
            # CDR ASSESSMENT
            # ------------------------------------------------

            gr.Markdown("### 3. Clinical Dementia Rating Scores")

            with gr.Row():

                manual_cdr_memory = gr.Dropdown(
                    choices=CDR_MEMORY_VALUES,
                    value=CDR_MEMORY_VALUES[0],
                    label="CDR Memory"
                )

                manual_cdr_orientation = gr.Dropdown(
                    choices=CDR_ORIENTATION_VALUES,
                    value=CDR_ORIENTATION_VALUES[0],
                    label="CDR Orientation"
                )

                manual_cdr_judgement = gr.Dropdown(
                    choices=CDR_JUDGEMENT_VALUES,
                    value=CDR_JUDGEMENT_VALUES[0],
                    label="CDR Judgement and Problem Solving"
                )

            with gr.Row():

                manual_cdr_community = gr.Dropdown(
                    choices=CDR_COMMUNITY_VALUES,
                    value=CDR_COMMUNITY_VALUES[0],
                    label="CDR Community Affairs"
                )

                manual_cdr_home = gr.Dropdown(
                    choices=CDR_HOME_VALUES,
                    value=CDR_HOME_VALUES[0],
                    label="CDR Home and Hobbies"
                )

                manual_cdr_care = gr.Dropdown(
                    choices=CDR_CARE_VALUES,
                    value=CDR_CARE_VALUES[0],
                    label="CDR Personal Care"
                )

            with gr.Row():

                manual_cdr_global = gr.Dropdown(
                    choices=CDR_GLOBAL_VALUES,
                    value=CDR_GLOBAL_VALUES[0],
                    label="Global CDR"
                )

                manual_cdr_sum_boxes = gr.Number(
                    label="CDR Sum of Boxes (0–9)",
                    value=1,
                    minimum=0,
                    maximum=9,
                    step=0.5
                )

            # ------------------------------------------------
            # MRI INPUT
            # ------------------------------------------------

            gr.Markdown("### 4. Structural MRI Input")

            manual_mri_source = gr.Radio(
                choices=[
                    "Use sample MRI",
                    "Upload processed MRI"
                ],
                value="Use sample MRI",
                label="MRI source"
            )

            with gr.Row():

                manual_sample_mri_rid = gr.Dropdown(
                    choices=participant_choices,
                    value=participant_rids[0],
                    label="Select a sample MRI participant"
                )

                manual_uploaded_mri = gr.File(
                    label=(
                        "Upload processed MRI "
                        "(.npy, expected shape 160 × 160 × 160)"
                    ),
                    file_types=[".npy"],
                    type="filepath"
                )

            manual_predict_button = gr.Button(
                "Run Manual Assessment",
                variant="primary"
            )

            # ------------------------------------------------
            # OUTPUTS
            # ------------------------------------------------

            manual_input_summary = gr.HTML()

            gr.Markdown("## Model Predictions")

            with gr.Row():

                manual_mri_result = gr.HTML(elem_classes=["result-card"])

                manual_cognitive_result = gr.HTML(elem_classes=["result-card"])

                manual_fusion_result = gr.HTML(elem_classes=["result-card"])

            gr.Markdown("## Model Probability Distributions")

            with gr.Row():

                manual_mri_probability = gr.Plot(elem_classes=["probability-chart"])

                manual_cognitive_probability = gr.Plot(elem_classes=["probability-chart"])

                manual_fusion_probability = gr.Plot(elem_classes=["probability-chart"])

            gr.Markdown("## MRI Input Preview")

            manual_mri_preview = gr.Plot(elem_classes=["mri-preview-chart"])

            manual_predict_button.click(
                fn=run_manual_prediction,
                inputs=[
                    manual_age,
                    manual_gender,
                    manual_education,
                    manual_mmse,
                    manual_delayed_recall,
                    manual_cdr_memory,
                    manual_cdr_orientation,
                    manual_cdr_judgement,
                    manual_cdr_community,
                    manual_cdr_home,
                    manual_cdr_care,
                    manual_cdr_global,
                    manual_cdr_sum_boxes,
                    manual_adas_total,
                    manual_adas_13,
                    manual_faq,
                    manual_mri_source,
                    manual_sample_mri_rid,
                    manual_uploaded_mri
                ],
                outputs=[
                    manual_input_summary,
                    manual_mri_result,
                    manual_cognitive_result,
                    manual_fusion_result,
                    manual_mri_probability,
                    manual_cognitive_probability,
                    manual_fusion_probability,
                    manual_mri_preview
                ]
            )

print("Step 17C completed.")

# ============================================================
# STEP 17D: CREATE ABOUT TAB
# ============================================================

with dementia_interface:

    with interface_tabs:

        with gr.Tab("About"):

            gr.Markdown(
                """
                ## About the Project

                **Project title:**
                AI-Based Multi-Modal Dementia Early Detection Tool

                **Data source:**
                Alzheimer's Disease Neuroimaging Initiative (ADNI)

                **Models used:**

                - MobileNetV2 for structural MRI classification
                - Random Forest for cognitive classification
                - Random Forest for multimodal fusion

                **Research classes:**

                - CN — Cognitively Normal
                - MCI — Mild Cognitive Impairment
                - AD — Alzheimer's Disease

                ### Interface Modes

                **Sample Patient**

                Uses the MRI scan and cognitive features of an
                existing matched ADNI participant.

                **Manual Entry**

                Allows cognitive and demographic values to be
                entered manually. The user may select a sample
                MRI or upload a compatible preprocessed MRI
                NumPy volume.

                ### Disclaimer

                This prototype is intended only for academic
                research and system demonstration.

                Its outputs do not represent medical advice,
                a confirmed clinical diagnosis or a treatment
                recommendation.
                """
            )

print("Step 17D completed.")

if __name__ == "__main__":
    dementia_interface.launch(share=False)
