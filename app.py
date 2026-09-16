
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import hashlib


# ==================================================
# PAGE SETTINGS
# ==================================================

st.set_page_config(
    page_title="Garbage AI | Detection & Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# SESSION STATE
# ==================================================

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Main header */

.hero {
    padding: 28px 20px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        rgba(46, 125, 50, 0.12),
        rgba(76, 175, 80, 0.04)
    );
    border: 1px solid rgba(76, 175, 80, 0.20);
    text-align: center;
    margin-bottom: 25px;
}

.hero-title {
    font-size: 44px;
    font-weight: 800;
    margin: 0;
}

.hero-subtitle {
    font-size: 18px;
    margin-top: 8px;
    opacity: 0.75;
}

/* Section titles */

.section-title {
    font-size: 25px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 12px;
}

/* Input card */

.input-card {
    padding: 20px;
    border-radius: 18px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    margin-bottom: 20px;
}

/* Status badge */

.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 12px;
}

/* Detection object */

.object-card {
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.22);
    margin-bottom: 12px;
}

/* Guide */

.guide-card {
    padding: 16px;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.18);
    margin-bottom: 10px;
}

/* Footer */

.footer {
    text-align: center;
    padding: 30px 10px 10px;
    opacity: 0.6;
    font-size: 14px;
}

.small-text {
    font-size: 14px;
    opacity: 0.7;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">

<div class="hero-title">
♻️ Garbage AI
</div>

<div class="hero-subtitle">
Intelligent Waste Detection, Classification & Segregation
</div>

<div class="small-text">
AI-powered computer vision system for identifying multiple waste objects
</div>

</div>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return YOLO("trash_model.pt")


model = load_model()


# ==================================================
# WASTE CATEGORIES
# ==================================================

waste_categories = {

    "Aluminium foil": "Recyclable / Dry Waste",
    "Bottle cap": "Recyclable / Dry Waste",
    "Bottle": "Recyclable / Dry Waste",
    "Broken glass": "Special Handling",
    "Can": "Recyclable / Dry Waste",
    "Carton": "Recyclable / Dry Waste",
    "Cigarette": "General Waste",
    "Cup": "Recyclable / Dry Waste",
    "Lid": "Recyclable / Dry Waste",
    "Other litter": "General Waste",
    "Other plastic": "Recyclable / Dry Waste",
    "Paper": "Recyclable / Dry Waste",
    "Plastic bag - wrapper": "Recyclable / Dry Waste",
    "Plastic container": "Recyclable / Dry Waste",
    "Pop tab": "Recyclable / Dry Waste",
    "Straw": "General Waste",
    "Styrofoam piece": "General Waste",
    "Unlabeled litter": "General Waste"
}


# ==================================================
# DISPLAY LABELS
# ==================================================

category_display = {

    "Recyclable / Dry Waste": "♻️ Recyclable / Dry Waste",
    "Wet / Organic Waste": "🟤 Wet / Organic Waste",
    "Special Handling": "⚠️ Special Handling",
    "General Waste": "🗑️ General Waste"
}


# ==================================================
# SEGREGATION GUIDE
# ==================================================

segregation_guide = {

    "Aluminium foil": {
        "category": "Recyclable / Dry Waste",
        "action": "Keep reasonably clean and place with recyclable metal materials."
    },

    "Bottle cap": {
        "category": "Recyclable / Dry Waste",
        "action": "Separate from food residue and recycle where accepted."
    },

    "Bottle": {
        "category": "Recyclable / Dry Waste",
        "action": "Empty and clean the bottle before placing it with recyclables."
    },

    "Broken glass": {
        "category": "Special Handling",
        "action": "Handle carefully, wrap securely and follow local glass-disposal guidance."
    },

    "Can": {
        "category": "Recyclable / Dry Waste",
        "action": "Empty the can and place it with metal recyclables."
    },

    "Carton": {
        "category": "Recyclable / Dry Waste",
        "action": "Keep the carton dry and place it with recyclable paper/cardboard."
    },

    "Cigarette": {
        "category": "General Waste",
        "action": "Ensure it is completely extinguished and dispose of it as general waste."
    },

    "Cup": {
        "category": "Recyclable / Dry Waste",
        "action": "Check the cup material and recycle only where that material is accepted."
    },

    "Lid": {
        "category": "Recyclable / Dry Waste",
        "action": "Separate from remaining contents and recycle where accepted."
    },

    "Other litter": {
        "category": "General Waste",
        "action": "Dispose of it according to local general-waste guidelines."
    },

    "Other plastic": {
        "category": "Recyclable / Dry Waste",
        "action": "Check the plastic type and local recycling rules before disposal."
    },

    "Paper": {
        "category": "Recyclable / Dry Waste",
        "action": "Keep paper dry and clean and place it with paper recyclables."
    },

    "Plastic bag - wrapper": {
        "category": "Recyclable / Dry Waste",
        "action": "Check local plastic-film collection rules before recycling."
    },

    "Plastic container": {
        "category": "Recyclable / Dry Waste",
        "action": "Empty and clean the container before recycling."
    },

    "Pop tab": {
        "category": "Recyclable / Dry Waste",
        "action": "Place it with recyclable metal materials."
    },

    "Straw": {
        "category": "General Waste",
        "action": "Dispose of it as general waste unless locally recyclable."
    },

    "Styrofoam piece": {
        "category": "General Waste",
        "action": "Usually dispose of it as general waste unless a local collection program accepts it."
    },

    "Unlabeled litter": {
        "category": "General Waste",
        "action": "Dispose of it according to local waste-management guidelines."
    }
}


# ==================================================
# DISPOSAL RECOMMENDATIONS
# ==================================================

recommendations = {

    "Aluminium foil": "Keep clean and send for metal recycling.",
    "Bottle cap": "Put with recyclable plastic.",
    "Bottle": "Empty, clean and send for recycling.",
    "Broken glass": "Handle carefully and wrap separately.",
    "Can": "Empty and send for metal recycling.",
    "Carton": "Keep dry and send for recycling.",
    "Cigarette": "Dispose as general waste.",
    "Cup": "Check the material and recycle where accepted.",
    "Lid": "Recycle where accepted.",
    "Other litter": "Dispose as general waste.",
    "Other plastic": "Recycle where the plastic type is accepted.",
    "Paper": "Keep dry and send for paper recycling.",
    "Plastic bag - wrapper": "Follow local rules for plastic-film recycling.",
    "Plastic container": "Empty, clean and recycle.",
    "Pop tab": "Put with metal recyclables.",
    "Straw": "Dispose as general waste unless locally recyclable.",
    "Styrofoam piece": "Usually dispose as general waste.",
    "Unlabeled litter": "Dispose as general waste."
}


# ==================================================
# INPUT SECTION
# ==================================================

st.markdown(
    '<div class="section-title">📷 Analyze Waste</div>',
    unsafe_allow_html=True
)

input_method = st.radio(
    "Choose an input method",
    ["Upload Image", "Use Camera"],
    horizontal=True
)

uploaded_file = None

if input_method == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a garbage image",
        type=["jpg", "jpeg", "png"],
        help="Upload an image containing one or more waste objects."
    )

else:

    st.caption(
        "Use your camera to capture an image containing waste objects."
    )

    uploaded_file = st.camera_input(
        "Take a picture"
    )


# ==================================================
# DETECTION
# ==================================================

if uploaded_file:

    image_bytes = uploaded_file.getvalue()

    image_id = hashlib.md5(image_bytes).hexdigest()

    image = Image.open(uploaded_file).convert("RGB")

    st.divider()

    # --------------------------------------------------
    # INPUT IMAGE
    # --------------------------------------------------

    st.markdown(
        '<div class="section-title">📷 Input Image</div>',
        unsafe_allow_html=True
    )

    image_column, info_column = st.columns(
        [2.2, 1]
    )

    with image_column:

        st.image(
            image,
            use_container_width=True
        )

    with info_column:

        st.markdown("### 🧠 AI Analysis")

        st.write(
            "The YOLO11 model will analyze the image and identify "
            "individual waste objects."
        )

        st.info(
            "Multiple objects can be detected in a single image."
        )


    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------

    with st.spinner(
        "🔍 AI is detecting waste objects..."
    ):

        results = model.predict(
            image,
            device="cpu",
            conf=0.45,
            imgsz=640
        )

    result = results[0]


    # ==================================================
    # NO DETECTIONS
    # ==================================================

    if len(result.boxes) == 0:

        st.warning(
            "⚠️ No waste objects were detected."
        )

        st.info(
            "Try a clearer image with better lighting and "
            "objects that are clearly visible."
        )


    # ==================================================
    # DETECTIONS FOUND
    # ==================================================

    else:

        # --------------------------------------------------
        # COLLECT DATA
        # --------------------------------------------------

        detected_classes = []

        detection_data = []


        for box in result.boxes:

            class_id = int(box.cls[0])

            confidence = float(
                box.conf[0]
            ) * 100

            class_name = result.names[class_id]

            category = waste_categories.get(
                class_name,
                "General Waste"
            )

            detected_classes.append(
                class_name
            )

            detection_data.append({

                "name": class_name,

                "confidence": confidence,

                "category": category

            })


        # --------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------

        if (
            st.session_state.last_image_id
            != image_id
        ):

            history_entry = {

                "objects":
                    detected_classes.copy(),

                "detections":
                    detection_data.copy(),

                "total_objects":
                    len(detected_classes)

            }

            st.session_state.detection_history.append(
                history_entry
            )

            st.session_state.last_image_id = image_id


        # ==================================================
        # RESULT TABS
        # ==================================================

        result_tab, segregation_tab, history_tab = st.tabs(
            [
                "🔍 Detection Results",
                "♻️ Segregation",
                "📈 Statistics & History"
            ]
        )


        # ==================================================
        # DETECTION RESULTS TAB
        # ==================================================

        with result_tab:

            st.markdown(
                '<div class="section-title">🔍 Detection Result</div>',
                unsafe_allow_html=True
            )

            annotated_image = result.plot()

            st.image(
                annotated_image,
                caption="AI Detected Objects",
                use_container_width=True
            )


            # --------------------------------------------------
            # SUMMARY CARDS
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">📊 Analysis Summary</div>',
                unsafe_allow_html=True
            )

            total_objects = len(
                detected_classes
            )

            unique_objects = len(
                set(detected_classes)
            )

            average_confidence = (
                sum(
                    item["confidence"]
                    for item in detection_data
                )
                / total_objects
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Objects Detected",
                    total_objects,
                    icon="🔍",
                    border=True
                )


            with col2:

                st.metric(
                    "Waste Types",
                    unique_objects,
                    icon="🏷️",
                    border=True
                )


            with col3:

                st.metric(
                    "Average Confidence",
                    f"{average_confidence:.2f}%",
                    icon="🎯",
                    border=True
                )


            # --------------------------------------------------
            # WASTE SUMMARY
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">♻️ Waste Summary</div>',
                unsafe_allow_html=True
            )

            waste_counts = Counter(
                detected_classes
            )

            summary_columns = st.columns(
                min(
                    len(waste_counts),
                    4
                )
            )


            for index, (
                waste_type,
                count
            ) in enumerate(
                waste_counts.items()
            ):

                with summary_columns[
                    index % len(summary_columns)
                ]:

                    st.metric(
                        waste_type,
                        count,
                        border=True
                    )


            # --------------------------------------------------
            # OBJECT DETAILS
            # --------------------------------------------------

            st.markdown(
                '<div class="section-title">📋 Detected Objects</div>',
                unsafe_allow_html=True
            )


            for number, item in enumerate(
                detection_data,
                start=1
            ):

                class_name = item["name"]

                confidence = item["confidence"]

                category = item["category"]

                recommendation = recommendations.get(
                    class_name,
                    "Check local waste disposal guidelines."
                )


                with st.container(
                    border=True
                ):

                    col1, col2, col3 = st.columns(
                        [1, 1.6, 2.4]
                    )


                    with col1:

                        st.markdown(
                            f"### #{number}"
                        )

                        st.write(
                            f"**{class_name}**"
                        )

                        st.caption(
                            f"Confidence: {confidence:.2f}%"
                        )


                    with col2:

                        st.markdown(
                            "**🗑️ Category**"
                        )

                        st.write(
                            category_display.get(
                                category,
                                "🗑️ General Waste"
                            )
                        )


                    with col3:

                        st.markdown(
                            "**♻️ Disposal Recommendation**"
                        )

                        st.write(
                            recommendation
                        )


        # ==================================================
        # SEGREGATION TAB
        # ==================================================

        with segregation_tab:

            st.markdown(
                '<div class="section-title">🗑️ Waste Segregation</div>',
                unsafe_allow_html=True
            )

            category_counts = Counter(
                item["category"]
                for item in detection_data
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "♻️ Recyclable / Dry",
                    category_counts.get(
                        "Recyclable / Dry Waste",
                        0
                    ),
                    border=True
                )

                st.metric(
                    "⚠️ Special Handling",
                    category_counts.get(
                        "Special Handling",
                        0
                    ),
                    border=True
                )


            with col2:

                st.metric(
                    "🟤 Wet / Organic",
                    category_counts.get(
                        "Wet / Organic Waste",
                        0
                    ),
                    border=True
                )

                st.metric(
                    "🗑️ General Waste",
                    category_counts.get(
                        "General Waste",
                        0
                    ),
                    border=True
                )


            st.markdown(
                '<div class="section-title">📖 Detailed Guide</div>',
                unsafe_allow_html=True
            )


            shown_types = set()


            for item in detection_data:

                class_name = item["name"]

                if class_name in shown_types:
                    continue

                shown_types.add(
                    class_name
                )

                guide = segregation_guide.get(
                    class_name,
                    {
                        "category":
                            "General Waste",

                        "action":
                            "Follow local waste-disposal guidelines."
                    }
                )


                with st.expander(
                    f"♻️ {class_name}"
                ):

                    st.markdown(
                        f"**Category:** "
                        f"{category_display.get(guide['category'], guide['category'])}"
                    )

                    st.markdown(
                        f"**What to do:** "
                        f"{guide['action']}"
                    )


        # ==================================================
        # HISTORY TAB
        # ==================================================

        with history_tab:

            history = (
                st.session_state.detection_history
            )


            if not history:

                st.info(
                    "No detection history yet."
                )

            else:

                # --------------------------------------------------
                # OVERALL STATISTICS
                # --------------------------------------------------

                total_images = len(history)

                total_objects_all = sum(
                    item["total_objects"]
                    for item in history
                )


                all_detections = []

                for entry in history:

                    all_detections.extend(
                        entry["detections"]
                    )


                all_waste_types = [
                    item["name"]
                    for item in all_detections
                ]


                all_categories = [
                    item["category"]
                    for item in all_detections
                ]


                unique_waste_types = len(
                    set(all_waste_types)
                )


                recyclable_count = all_categories.count(
                    "Recyclable / Dry Waste"
                )

                wet_count = all_categories.count(
                    "Wet / Organic Waste"
                )

                special_count = all_categories.count(
                    "Special Handling"
                )

                general_count = all_categories.count(
                    "General Waste"
                )


                # --------------------------------------------------
                # STATISTICS CARDS
                # --------------------------------------------------

                st.markdown(
                    '<div class="section-title">📊 Overall Statistics</div>',
                    unsafe_allow_html=True
                )


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "🖼️ Images",
                        total_images,
                        border=True
                    )


                with col2:

                    st.metric(
                        "🔍 Objects",
                        total_objects_all,
                        border=True
                    )


                with col3:

                    st.metric(
                        "♻️ Recyclable",
                        recyclable_count,
                        border=True
                    )


                with col4:

                    st.metric(
                        "🗑️ General Waste",
                        general_count,
                        border=True
                    )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "🏷️ Waste Types",
                        unique_waste_types,
                        border=True
                    )


                with col2:

                    st.metric(
                        "⚠️ Special",
                        special_count,
                        border=True
                    )


                with col3:

                    st.metric(
                        "🟤 Wet / Organic",
                        wet_count,
                        border=True
                    )


                # --------------------------------------------------
                # WASTE TYPE CHART
                # --------------------------------------------------

                st.markdown(
                    '<div class="section-title">📊 Waste Type Statistics</div>',
                    unsafe_allow_html=True
                )


                waste_statistics = Counter(
                    all_waste_types
                )


                if waste_statistics:

                    st.bar_chart(
                        dict(waste_statistics),
                        horizontal=True
                    )


                # --------------------------------------------------
                # SEGREGATION CHART
                # --------------------------------------------------

                st.markdown(
                    '<div class="section-title">🗑️ Segregation Statistics</div>',
                    unsafe_allow_html=True
                )


                segregation_statistics = {

                    "Recyclable / Dry":
                        recyclable_count,

                    "Wet / Organic":
                        wet_count,

                    "Special Handling":
                        special_count,

                    "General Waste":
                        general_count
                }


                st.bar_chart(
                    segregation_statistics,
                    horizontal=True
                )


                # --------------------------------------------------
                # HISTORY
                # --------------------------------------------------

                st.markdown(
                    '<div class="section-title">📜 Detection History</div>',
                    unsafe_allow_html=True
                )


                for index, entry in enumerate(
                    reversed(history),
                    start=1
                ):

                    image_number = (
                        len(history) - index + 1
                    )


                    objects = Counter(
                        entry["objects"]
                    )


                    object_text = ", ".join(
                        f"{name} ({count})"
                        for name, count
                        in objects.items()
                    )


                    with st.expander(
                        f"🖼️ Detection {image_number}  •  "
                        f"{entry['total_objects']} object(s)"
                    ):

                        st.write(
                            f"**Detected objects:** "
                            f"{object_text}"
                        )


                        detection_categories = Counter(
                            item["category"]
                            for item in entry["detections"]
                        )


                        st.write(
                            "**Segregation:**"
                        )


                        for category, count in (
                            detection_categories.items()
                        ):

                            st.write(
                                f"• "
                                f"{category_display.get(category, category)}"
                                f": {count}"
                            )


                st.divider()


                if st.button(
                    "🗑️ Clear Detection History",
                    type="secondary"
                ):

                    st.session_state.detection_history = []

                    st.session_state.last_image_id = None

                    st.rerun()


# ==================================================
# GENERAL GUIDE
# ==================================================

st.divider()

with st.expander(
    "📚 General Waste Segregation Guide"
):

    st.markdown("""
### ♻️ Recyclable / Dry Waste

Examples:
- Paper
- Cardboard
- Metal cans
- Some plastic containers
- Bottles

Keep recyclable materials reasonably clean and dry.

### 🟤 Wet / Organic Waste

Examples:
- Food scraps
- Vegetable waste
- Fruit waste
- Other biodegradable organic materials

Keep wet waste separate from dry recyclable materials.

### ⚠️ Special Handling

Examples:
- Broken glass
- Certain hazardous materials
- Items requiring special collection

Handle these materials carefully and follow local disposal guidance.

### 🗑️ General Waste

Materials that cannot be practically recycled through the available
local system can generally go into the appropriate general-waste stream.
""")


# ==================================================
# ABOUT PROJECT
# ==================================================

st.divider()

with st.expander(
    "ℹ️ About This Project"
):

    st.write("""
Garbage Detection and Classification is an AI-based
computer vision application designed to detect multiple
waste objects in a single image.

### System Features

• Multi-object waste detection  
• Bounding box visualization  
• Confidence scores  
• Waste counting  
• Waste classification  
• Waste segregation  
• Disposal recommendations  
• Camera input  
• Image upload  
• Detection history  
• Statistical analysis  

### AI Model

The application uses a YOLO11 object detection model
trained for waste detection.

The current model recognizes 18 waste categories.
""")


# ==================================================
# FOOTER
# ==================================================

st.markdown("""
<div class="footer">

♻️ <b>Garbage AI</b><br>

Garbage Detection & Classification | Final Year Project

</div>
""", unsafe_allow_html=True)
