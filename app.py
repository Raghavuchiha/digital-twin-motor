
import os
import joblib
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Auto Twin",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Auto Twin")
st.caption("Motor Fault Detection and Digital Twin Monitoring")

# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

@st.cache_resource
def load_models():
    fault_model = joblib.load("models/fault_model.joblib")
    twin_model = joblib.load("models/digital_twin.joblib")
    return fault_model, twin_model


try:
    fault, twin = load_models()
except Exception as e:
    st.error(f"Could not load saved models: {e}")
    st.info(
        "Check that both model files exist inside the models folder."
    )
    st.stop()


# --------------------------------------------------
# INITIALIZE SESSION STATE
# --------------------------------------------------

if "row_index" not in st.session_state:
    st.session_state.row_index = 0

if "playing" not in st.session_state:
    st.session_state.playing = False

if "last_refresh_count" not in st.session_state:
    st.session_state.last_refresh_count = 0

if "loaded_path" not in st.session_state:
    st.session_state.loaded_path = ""


# --------------------------------------------------
# CSV INPUT
# --------------------------------------------------

st.sidebar.header("Data Source")

csv_path = st.sidebar.text_input(
    "CSV file path",
    placeholder=r"D:\sem-7\mini-project\your_data.csv"
).strip().strip('"')

if not csv_path:
    st.info("Enter the full path of your motor dataset in the sidebar.")
    st.stop()

if not os.path.isfile(csv_path):
    st.error("CSV file not found. Check the file path.")
    st.stop()


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


try:
    df = load_csv(csv_path)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()


if df.empty:
    st.error("The CSV file contains no rows.")
    st.stop()


# Reset replay when the input path changes
if st.session_state.loaded_path != csv_path:
    st.session_state.loaded_path = csv_path
    st.session_state.row_index = 0
    st.session_state.playing = False
    st.session_state.last_refresh_count = 0


# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = (
    set(fault["features"])
    | set(twin["features"])
    | {"speed_rpm", "Current", "Voltage", "temp_mot"}
)

missing_columns = required_columns - set(df.columns)

if missing_columns:
    st.error(
        "The CSV is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()


# --------------------------------------------------
# PLAYBACK SETTINGS
# --------------------------------------------------

st.sidebar.header("Replay Settings")

speed = st.sidebar.selectbox(
    "Playback speed",
    options=[0.5, 1, 2, 5],
    index=1,
    format_func=lambda value: f"{value} row(s) per second"
)

interval_ms = int(1000 / speed)


# --------------------------------------------------
# REPLAY CONTROLS
# --------------------------------------------------

st.subheader("CSV Replay Controls")

start_col, pause_col, restart_col = st.columns(3)

with start_col:
    if st.button(
        "▶ Start / Resume",
        use_container_width=True
    ):
        st.session_state.playing = True
        st.rerun()

with pause_col:
    if st.button(
        "⏸ Pause",
        use_container_width=True
    ):
        st.session_state.playing = False
        st.rerun()

with restart_col:
    if st.button(
        "↺ Restart",
        use_container_width=True
    ):
        st.session_state.row_index = 0
        st.session_state.playing = False
        st.session_state.last_refresh_count = 0
        st.rerun()


# Advance one row on each automatic refresh.
# Do not advance when paused or replay has ended.

if st.session_state.playing:
    refresh_count = st_autorefresh(
        interval=interval_ms,
        key="auto_twin_refresh"
    )

    if refresh_count > st.session_state.last_refresh_count:
        st.session_state.last_refresh_count = refresh_count

        # Advance only after a refresh tick, not immediately
        # when the user presses Start.
        if refresh_count > 0:
            st.session_state.row_index += 1

        if st.session_state.row_index >= len(df):
            st.session_state.row_index = len(df) - 1
            st.session_state.playing = False

        st.rerun()


# --------------------------------------------------
# CURRENT SENSOR READING
# --------------------------------------------------

idx = st.session_state.row_index
row = df.iloc[idx]

st.divider()

status_col, progress_col = st.columns([1, 2])

with status_col:
    if st.session_state.playing:
        st.success("Replay running")
    elif idx == len(df) - 1:
        st.info("Replay reached the final row")
    else:
        st.warning("Replay paused")

with progress_col:
    st.progress((idx + 1) / len(df))
    st.caption(f"Reading {idx + 1} of {len(df)}")


# --------------------------------------------------
# SENSOR PANEL
# --------------------------------------------------

st.header("Live Sensor Readings")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Motor RPM",
    f"{row['speed_rpm']:.2f}"
)

c2.metric(
    "Current",
    f"{row['Current']:.3f}"
)

c3.metric(
    "Voltage",
    f"{row['Voltage']:.2f}"
)

c4.metric(
    "Recorded Temperature",
    f"{row['temp_mot']:.2f}"
)


# --------------------------------------------------
# FAULT DETECTION
# --------------------------------------------------

st.header("1. Fault Detection")

fault_features = fault["features"]

fault_input = pd.DataFrame(
    [[row[col] for col in fault_features]],
    columns=fault_features
)

fault_score = None
is_anomaly = None

if fault_input.isna().any().any():
    st.warning(
        "Fault detection unavailable: a required sensor "
        "feature is missing for this reading."
    )
else:
    try:
        scaled_input = fault["scaler"].transform(fault_input)

        fault_score = float(
            -fault["model"].score_samples(scaled_input)[0]
        )

        is_anomaly = fault_score > fault["threshold"]

        score_col, threshold_col = st.columns(2)

        score_col.metric(
            "Anomaly Score",
            f"{fault_score:.4f}"
        )

        threshold_col.metric(
            "Detection Threshold",
            f"{fault['threshold']:.4f}"
        )

        if is_anomaly:
            st.error(
                "⚠️ Abnormal motor behaviour detected by the model."
            )
        else:
            st.success(
                "No anomaly detected by the model for this reading."
            )

    except Exception as e:
        st.error(f"Fault detection failed: {e}")


# --------------------------------------------------
# DIGITAL TWIN TEMPERATURE PREDICTION
# --------------------------------------------------

st.header("2. Digital Twin")

twin_features = twin["features"]

twin_input = pd.DataFrame(
    [[row[col] for col in twin_features]],
    columns=twin_features
)

predicted_temp = None

if twin_input.isna().any().any():
    st.warning(
        "Temperature prediction unavailable: a required "
        "sensor feature is missing."
    )
else:
    try:
        predicted_temp = float(
            twin["model"].predict(twin_input)[0]
        )

        actual_temp = float(row["temp_mot"])

        pred_col, actual_col, error_col = st.columns(3)

        pred_col.metric(
            "Predicted Temperature",
            f"{predicted_temp:.2f}"
        )

        actual_col.metric(
            "Recorded Temperature",
            f"{actual_temp:.2f}"
        )

        error_col.metric(
            "Absolute Difference",
            f"{abs(predicted_temp - actual_temp):.2f}"
        )

        st.caption(
            "The difference compares a prediction with the "
            "recorded temperature for this same dataset row. "
            "It is not a future-temperature forecast."
        )

    except Exception as e:
        st.error(f"Digital Twin prediction failed: {e}")


# --------------------------------------------------
# REPLAY HISTORY
# --------------------------------------------------

st.header("3. Recent Sensor History")

start_index = max(0, idx - 49)
history = df.iloc[start_index:idx + 1]

chart_columns = [
    col for col in ["speed_rpm", "Current", "Voltage", "temp_mot"]
    if col in history.columns
]

if chart_columns:
    st.line_chart(history[chart_columns])

with st.expander("View current raw sensor row"):
    st.dataframe(
        row.to_frame(name="Value"),
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Auto Twin prototype | CSV replay mode. "
    "Predictions and anomaly scores are estimates, not "
    "guaranteed fault diagnoses or safety controls. "
    "This dashboard does not control the physical motor."
)