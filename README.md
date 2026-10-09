
# Auto Twin — AI-Powered Motor Digital Twin

An AI-powered software prototype for industrial motor monitoring, anomaly detection, temperature estimation, and sensor-data visualization.

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Features](#4-features)
5. [System Architecture](#5-system-architecture)
6. [Technology Stack](#6-technology-stack)
7. [Project Structure](#7-project-structure)
8. [How the System Works](#8-how-the-system-works)
9. [Machine Learning Models](#9-machine-learning-models)
10. [Dataset Description](#10-dataset-description)
11. [Installation and Setup](#11-installation-and-setup)
12. [Running the Application](#12-running-the-application)
13. [Training the Models](#13-training-the-models)
14. [Understanding the Dashboard](#14-understanding-the-dashboard)
15. [Model Evaluation](#15-model-evaluation)
16. [Troubleshooting](#16-troubleshooting)
17. [Current Limitations](#17-current-limitations)
18. [Future Improvements](#18-future-improvements)
19. [Contributing](#19-contributing)
20. [Disclaimer](#20-disclaimer)

---

## 1. Project Overview

**Auto Twin** is a software-based digital twin prototype designed to monitor the behavior of an industrial motor using sensor data and machine learning.

A digital twin is a virtual representation of a physical asset. It uses data from the asset to help understand its behavior, monitor its condition, and evaluate possible operating scenarios.

Auto Twin applies this concept to an industrial motor by combining:

- Sensor data such as current, voltage, rotational speed, and temperature.
- Vibration measurements for detecting unusual motor behavior.
- An Isolation Forest model for anomaly detection.
- A Random Forest regression model for motor-temperature estimation.
- A Streamlit dashboard for replaying sensor records and visualizing results.

The current implementation uses historical CSV data to simulate a stream of sensor readings. It is a software prototype and does not require a physical motor or connected sensors to run.

**Repository:** https://github.com/Raghavuchiha/digital-twin-motor

## 2. Problem Statement

Industrial motors can experience abnormal operating conditions because of excessive load, unusual vibration, temperature changes, electrical variations, or mechanical problems.

Identifying these conditions early can help maintenance teams investigate potential issues before they become more serious.

Traditional monitoring systems may depend on fixed thresholds or manual inspection. Such approaches can struggle to identify unusual combinations of sensor readings.

Auto Twin explores a data-driven approach in which machine learning models analyze motor sensor readings, identify anomalous behavior, and estimate motor temperature.

The dashboard makes the results easier to understand by presenting sensor measurements, model outputs, and recent readings in one place.

## 3. Project Objectives

The main objectives are:

1. **Motor monitoring:** Display important motor sensor readings.
2. **Anomaly detection:** Identify sensor patterns that differ from normal operating behavior.
3. **Temperature estimation:** Estimate motor temperature from selected operating parameters.
4. **Sensor replay:** Simulate a live monitoring experience using historical CSV records.
5. **Visualization:** Display current readings and recent trends through a dashboard.
6. **Modular development:** Keep data processing, model training, and dashboard logic in separate Python files where practical.
7. **Extensibility:** Provide a foundation for future integration with real-time sensor data and a decision-support module.

## 4. Features

### 4.1 Sensor monitoring

The application works with motor measurements such as:

- Rotational speed in RPM.
- Electrical current.
- Electrical voltage.
- Ambient temperature.
- Motor temperature.
- First-order vibration.
- Fifth-order vibration.

Additional columns in the dataset may be used for feature engineering, evaluation, or analysis.

### 4.2 Anomaly detection

The Isolation Forest model learns patterns from data considered normal during training.

Incoming sensor readings are transformed using a saved scaler and evaluated by the trained model. The resulting anomaly score is compared with a configured threshold to determine whether the reading should be flagged.

This is anomaly detection, not a diagnosis of a specific physical fault.

### 4.3 Motor-temperature estimation

A Random Forest regression model estimates motor temperature using:

- Rotational speed.
- Electrical current.
- Electrical voltage.
- Ambient temperature.

The estimated temperature can be compared with the recorded motor temperature to help understand model behavior.

### 4.4 CSV-based sensor replay

Instead of requiring a physical sensor, the dashboard replays rows from a CSV file.

Playback controls support starting or resuming, pausing, restarting, and changing the replay speed. This allows the team to test the dashboard with repeatable data.

### 4.5 Interactive dashboard

The Streamlit interface brings together sensor readings, anomaly-detection output, temperature estimation, and recent history.

---

## 5. System Architecture

The current system follows this high-level workflow:

```text
                 Historical Motor Dataset
                            |
                            v
                    CSV Data Loading
                            |
                            v
                   Sensor Row Selection
                            |
                            v
                   Feature Preparation
                            |
              +-------------+-------------+
              |                           |
              v                           v
      Isolation Forest           Random Forest
      Anomaly Detection          Temperature Estimation
              |                           |
              v                           v
      Anomaly Score              Estimated Temperature
      and Threshold                        |
              |                           |
              +-------------+-------------+
                            |
                            v
                    Streamlit Dashboard
                            |
                            v
               Metrics, Alerts and Trends
```

### Main components

| Component | Responsibility |
|---|---|
| CSV dataset | Supplies historical motor measurements |
| `model.py` | Trains and saves the anomaly-detection model |
| `digital_twin.py` | Trains and saves the temperature-regression model |
| `models/` | Stores the trained model artifacts |
| `app.py` | Loads the models and runs the interactive dashboard |
| `requirements.txt` | Lists the Python packages required to run the application |

The dashboard uses previously trained models. It does not need to retrain them every time the application starts.

---

## 6. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Loading and processing tabular sensor data |
| Scikit-learn | Training and running machine learning models |
| Isolation Forest | Detecting anomalous sensor patterns |
| Random Forest Regressor | Estimating motor temperature |
| StandardScaler | Scaling features for anomaly detection |
| Joblib | Saving and loading trained model artifacts |
| Streamlit | Building the interactive web dashboard |
| Streamlit Autorefresh | Supporting periodic dashboard refreshes |
| Git and GitHub | Version control and team collaboration |

---

## 7. Project Structure

The following describes the main files and directories used by the project.

```text
digital-twin-motor/
│
├── app.py
│   └── Main Streamlit dashboard
│
├── model.py
│   └── Trains the Isolation Forest anomaly detector
│
├── digital_twin.py
│   └── Trains the Random Forest temperature estimator
│
├── requirements.txt
│   └── Python dependency list
│
├── .gitignore
│   └── Files and directories excluded from Git
│
├── README.md
│   └── Project documentation
│
├── models/
│   ├── fault_model.joblib
│   │   └── Saved anomaly-detection model and preprocessing information
│   │
│   └── digital_twin.joblib
│       └── Saved temperature-regression model and feature information
│
├── outputs_stage2/
│   └── Saved evaluation plots, when present
│
└── dataset.csv
    └── Example name only; use the actual CSV filename supplied
        with the project
```

**Important:** The dataset filename shown above is illustrative. Use the actual dataset filename and path configured in `app.py` or the training scripts. The `outputs_stage2/` directory may be excluded by `.gitignore`, so it may not be present in a fresh clone.

### 7.1 `app.py` — Main application

This is the entry point for the dashboard.

Its responsibilities include:

- Loading the saved models from the `models/` directory.
- Reading sensor data from a CSV file.
- Replaying the records at a configurable speed.
- Displaying sensor measurements and recent history.
- Running anomaly detection on the selected sensor reading.
- Estimating motor temperature.
- Presenting the outputs in a Streamlit interface.

The application uses Streamlit caching for model loading so that the saved artifacts do not need to be reloaded on every dashboard refresh.

Run this file using Streamlit rather than running it directly with Python.

### 7.2 `model.py` — Anomaly-detection training

This file trains the Isolation Forest model.

Its main responsibilities are:

1. Load and prepare the training data.
2. Select the configured sensor and engineered features.
3. Fit the scaler and anomaly detector using the normal training subset.
4. Calculate anomaly scores and determine a decision threshold.
5. Save the trained model, scaler, threshold, and feature names.

The saved artifact is:

`models/fault_model.joblib`

The artifact contains a dictionary with the following information:

| Key | Purpose |
|---|---|
| `scaler` | Transforms sensor features before scoring |
| `model` | Trained Isolation Forest model |
| `threshold` | Cutoff used to flag anomalous readings |
| `features` | Ordered feature names expected by the model |

The dashboard must use the same feature names and ordering that were used during training.

### 7.3 `digital_twin.py` — Temperature-estimation training

This file trains a Random Forest regression model to estimate motor temperature.

The configured input features are:

- `speed_rpm`
- `Current`
- `Voltage`
- `temp_amb`

The target variable is `temp_mot`.

The script splits the data into training and testing subsets, trains the regressor, evaluates its predictions, and saves the trained model.

The saved artifact is:

`models/digital_twin.joblib`

The artifact contains:

| Key | Purpose |
|---|---|
| `model` | Trained Random Forest regressor |
| `features` | Ordered list of required input features |

### 7.4 `requirements.txt` — Dependencies

This file lists the Python libraries needed by the project.

Install them using:

```bash
pip install -r requirements.txt
```

### 7.5 `models/` — Trained model artifacts

The `models/` directory contains the fitted models required by the dashboard.

These are serialized Python objects created with Joblib. They are not source-code files and should not be edited manually.

If the saved models are missing, the dashboard will need the models to be regenerated using the corresponding training scripts.

**Security note:** Only load Joblib files from trusted sources because loading serialized Python objects can execute unsafe code.

---

## 8. How the System Works

### Step 1: Load sensor data

The application reads a CSV dataset containing historical motor readings.

Each row represents a recorded observation. The application selects a row during replay and uses its sensor values for the current dashboard update.

### Step 2: Prepare the model inputs

The sensor values are arranged into the feature columns expected by each model.

For anomaly detection, the input must be transformed with the saved scaler.

For temperature estimation, the application selects the required regression features in the correct order.

### Step 3: Detect anomalies

The scaled input is evaluated by the Isolation Forest model.

The application calculates an anomaly score and compares it with the configured threshold.

If the score crosses the threshold in the configured direction, the reading is flagged as anomalous.

The threshold is specific to the trained model and dataset. It is not a universal motor-safety limit.

### Step 4: Estimate motor temperature

The temperature model uses speed, current, voltage, and ambient temperature to estimate `temp_mot`.

This output is a regression estimate, not a physical temperature measurement.

In the current implementation, the estimate is based on the selected row's operating inputs. It is **not yet a future-temperature forecast**.

### Step 5: Display the results

The Streamlit dashboard displays the selected sensor readings and model outputs, along with recent history and playback controls.

The CSV replay makes it possible to test the monitoring workflow without a live sensor connection.

---

## 9. Machine Learning Models

### 9.1 Isolation Forest for anomaly detection

Isolation Forest is an unsupervised anomaly-detection algorithm. It identifies observations that are easier to isolate from the rest of the data.

In this project, the model is trained using observations designated as normal.

The general process is:

1. Select the configured sensor features.
2. Fit `StandardScaler` on the normal training data.
3. Transform the sensor features.
4. Fit Isolation Forest to the transformed normal data.
5. Calculate training anomaly scores.
6. Determine a threshold using the training-score distribution.
7. Apply the same preprocessing and scoring logic to new readings.

The selected features include:

- `Current`
- `Voltage`
- `temp_mot`
- `1x_Vibration`
- `5x_Vibration`
- `temp_mot_d5`
- `vib1x_d5`

The exact feature list is stored in the saved model artifact.

The engineered features `temp_mot_d5` and `vib1x_d5` represent changes over a five-step interval in the dataset's sequence. Their values depend on how those features were created during preprocessing.

**Interpretation:** An anomalous reading means the sensor pattern differs from the learned normal pattern. It does not automatically mean the motor has failed, and the model does not identify a specific fault type.

### 9.2 Random Forest Regressor for temperature estimation

Random Forest Regression combines predictions from multiple decision trees to estimate a continuous value.

The model in this project predicts motor temperature using four inputs:

\[
X = [\text{speed\_rpm},\text{Current},\text{Voltage},\text{temp\_amb}]
\]

The target is:

\[
y = \text{temp\_mot}
\]

The workflow is:

1. Load the dataset.
2. Select the input features and target.
3. Split the observations into training and testing sets.
4. Train the Random Forest regressor.
5. Generate predictions for the test subset.
6. Evaluate the predictions using MAE and R².
7. Save the model and input feature list.

The model learns relationships present in the training dataset. Its performance depends on the quality and coverage of that dataset.

### 9.3 Why use two separate models?

The models perform different tasks.

| Model | Task | Output |
|---|---|---|
| Isolation Forest | Anomaly detection | Anomaly score and threshold-based flag |
| Random Forest Regressor | Temperature estimation | Estimated motor temperature |

One model identifies unusual sensor patterns; the other estimates a numeric temperature value.

Neither model alone provides a confirmed mechanical diagnosis or a guaranteed corrective action.

---

## 10. Dataset Description

The dataset contains motor-operating observations and engineered features. The available columns include sensor measurements, vibration features, sequence information, and labels.

### Important columns

| Column | Description |
|---|---|
| `speed_rpm` | Motor rotational speed in revolutions per minute |
| `speed_Hz` | Speed-related frequency measurement, as defined by the dataset |
| `Current` | Recorded electrical current |
| `Voltage` | Recorded electrical voltage |
| `temp_amb` | Ambient temperature |
| `temp_mot` | Recorded motor temperature |
| `1x_Vibration` | First-order vibration feature |
| `5x_Vibration` | Fifth-order vibration feature |
| `temp_mot_d5` | Engineered motor-temperature change over five sequence steps |
| `vib1x_d5` | Engineered first-order vibration change over five sequence steps |
| `time` | Time-related dataset field |
| `run` | Identifier for a recorded run |
| `t_sec` | Time in seconds, according to the dataset definition |
| `label` | Dataset-provided condition label |

The dataset also includes additional electrical, vibration, frequency-band, and engineered features.

The label values include `normal`, `stress`, and `failed`.

These labels are useful for evaluating model behavior. However, the current Isolation Forest implementation is not a three-class classifier: it produces anomaly scores and a threshold-based flag.

### Data preparation considerations

When using another dataset, verify:

- The CSV includes all required columns.
- Column names match the model's feature list.
- Numeric columns contain valid numeric values.
- Missing values are handled appropriately.
- The units and meanings of sensor columns are consistent.
- Sequence-dependent features are calculated correctly.
- Training and testing data are separated in a way that avoids leakage between related observations.

If you use a different dataset, the existing saved models may not be appropriate without retraining and validation.

---

## 11. Installation and Setup

### Prerequisites

Install the following before running the application:

- Python 3.11 or a compatible Python version.
- Git.
- Visual Studio Code or another Python editor.
- A terminal such as PowerShell.
- The project dataset and trained model artifacts.

### Step 1: Clone the repository

Open PowerShell or the VS Code terminal and run:

```bash
git clone https://github.com/Raghavuchiha/digital-twin-motor.git
cd digital-twin-motor
```

### Step 2: Create a virtual environment (optional but recommended)

A virtual environment isolates this project's packages from other Python projects.

On Windows:

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If activation is blocked by the PowerShell execution policy, use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

Creating a virtual environment is optional. It is not a project requirement.

### Step 3: Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify the required files

Before launching the application, confirm that:

- `app.py` exists.
- `models/fault_model.joblib` exists.
- `models/digital_twin.joblib` exists.
- The dataset exists at the path expected by the application.
- All required packages have been installed.

### Step 5: Start the dashboard

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

Do not launch the dashboard using `python app.py`; the application is intended to run through Streamlit.

---

## 12. Running the Application

### Normal workflow

1. Open the repository in VS Code.
2. Open the terminal in the project root.
3. Activate the virtual environment if one is being used.
4. Install dependencies if they have not already been installed.
5. Verify the model artifacts and CSV path.
6. Run `streamlit run app.py`.
7. Use the dashboard's controls to start, pause, resume, restart, or change replay speed.
8. Observe the sensor readings, anomaly output, estimated temperature, and recent history.

### Replay speed

The dashboard supports the following replay speeds:

| Speed | Behavior |
|---|---|
| 0.5 rows/second | Slower replay |
| 1 row/second | One row per second |
| 2 rows/second | Faster replay |
| 5 rows/second | Fast replay |

These settings control how quickly records are replayed by the application. They do not change the actual sampling rate of a physical motor.

### Using another CSV file

If you want to use another dataset:

1. Inspect the expected CSV path in `app.py`.
2. Check the feature columns used by both models.
3. Ensure the CSV contains compatible data.
4. Select or configure the new CSV path using the application's supported input.
5. Restart the replay and verify that the rows are being read correctly.

A different CSV can be used for replay only if its structure is compatible with the application. Using a new dataset for model training is a separate process.

---

## 13. Training the Models

The dashboard loads existing model artifacts. Model training is performed separately.

### Train the anomaly detector

From the project root:

```bash
python model.py
```

After successful training, verify that the expected artifact exists:

```text
models/fault_model.joblib
```

### Train the temperature estimator

Run:

```bash
python digital_twin.py
```

After successful training, verify that the expected artifact exists:

```text
models/digital_twin.joblib
```

### Important training notes

- Run each script from the project root unless the script specifies otherwise.
- Make sure the training dataset is available at the path expected by the script.
- Inspect the feature lists and target column before changing the dataset.
- Keep the saved model structure compatible with the code in `app.py`.
- Retrain and evaluate the models whenever the dataset or feature engineering changes.
- Do not assume that a model trained on one motor or set of operating conditions will generalize to every motor.

The order of training is not important for these two independent models, provided their data and dependencies are available.

---

## 14. Understanding the Dashboard

The dashboard combines the outputs of the data replay, anomaly detector, and temperature estimator.

### Sensor measurements

These represent values read from the currently selected CSV record. They are historical dataset measurements, not readings from a connected physical motor.

### Anomaly score and flag

The anomaly score measures how the selected observation is evaluated by the trained Isolation Forest model.

The threshold determines whether the application flags the observation as anomalous. A flagged reading should be interpreted as a reason to investigate, not proof of a particular fault.

### Estimated motor temperature

This is the Random Forest model's estimate based on the selected row's speed, current, voltage, and ambient temperature.

The current implementation estimates the recorded row's motor temperature from its other inputs. It does not yet forecast the temperature several seconds or minutes into the future.

### Recent history

The recent-history visualization helps the user inspect how sensor values change as the application moves through the CSV records.

### Playback controls

Playback controls let the user inspect the same dataset at different speeds and repeat the simulation.

---

## 15. Model Evaluation

The current temperature-regression experiment reported the following results on a randomly selected test subset:

| Metric | Reported result |
|---|---:|
| Training observations | 301 |
| Testing observations | 76 |
| Mean Absolute Error (MAE) | Approximately 0.967 |
| R² score | Approximately 0.983 |

### What do these metrics mean?

**Mean Absolute Error (MAE)** measures the average absolute difference between the actual target and the predicted target.

An MAE of approximately 0.967 means the average absolute prediction error was about 0.967 in the target temperature's units.

**R² score** describes how well the model explains the variation in the test target relative to a baseline that predicts the target mean.

An R² score of approximately 0.983 indicates a strong fit on that test subset.

These numbers are results from the current experiment, not a guarantee of performance on unseen motors or operating conditions.

### Important validation limitation

A random train/test split can place observations from the same motor run in both subsets. If nearby readings are highly correlated, the test score may overstate performance on a completely new run.

A stronger evaluation should train on one set of motor runs and test on a separate, previously unseen run.

The anomaly detector has also been tested across different runs. Some test configurations showed substantial false-positive rates on normal observations, while stress and failed observations were often flagged.

This is useful preliminary evidence, but the anomaly detector still requires careful threshold tuning and validation on independent runs.

### Recommended evaluation improvements

- Use run-wise or time-wise train/test separation.
- Report MAE and R² on unseen runs.
- Measure false-positive and false-negative rates for anomaly detection.
- Evaluate the model on different loads and operating conditions.
- Check for data leakage from engineered features.
- Document the dataset and evaluation split used for each reported result.

---

## 16. Troubleshooting

### Error: Model file not found

Example:

```text
FileNotFoundError: models/fault_model.joblib
```

**Possible cause:** The saved model is missing, or the application is being launched from the wrong directory.

**Solution:**

1. Open the project root.
2. Check that both `.joblib` files exist inside `models/`.
3. If an artifact is missing, run its corresponding training script.
4. Launch the dashboard from the project root.

### Error: CSV file not found

**Possible cause:** The dataset path configured in the application does not match the location of the CSV file.

**Solution:** Check the path in `app.py`, ensure the file exists, and select or configure the correct path.

### Error: Missing feature columns

**Possible cause:** The CSV does not contain all the columns required by the model.

**Solution:** Compare the CSV column names with the model's saved feature list and correct the input schema.

Do not silently substitute unrelated columns for missing features.

### Error: No module named `streamlit`

**Solution:**

```bash
pip install -r requirements.txt
```

If multiple Python installations are present, make sure the packages are installed in the same Python environment used to launch Streamlit.

### Error: Streamlit dashboard does not start

Launch the app with:

```bash
streamlit run app.py
```

Running `python app.py` directly does not initialize the normal Streamlit application context.

### Error: Git changes are not appearing on GitHub

Check the current status:

```bash
git status
```

Then add and commit the intended changes:

```bash
git add .
git commit -m "Update project documentation"
git push
```

If a required dataset or model is missing from the repository, check `.gitignore`, file size limits, and whether the file was included in the commit.

---

## 17. Current Limitations

Auto Twin is a prototype intended for development and experimentation.

The current implementation has the following limitations:

1. **Historical-data replay:** The dashboard simulates sensor updates from CSV records rather than reading from live hardware.
2. **Anomaly detection, not fault classification:** The Isolation Forest flags unusual patterns but does not identify a confirmed failure type.
3. **Current-row temperature estimation:** The regression model estimates motor temperature for the selected record; it is not yet a future-temperature forecasting model.
4. **Dataset dependence:** Model performance depends on the quality and representativeness of the training data.
5. **Generalization:** Performance on a random test split may not represent performance on a completely new motor run.
6. **Threshold sensitivity:** Anomaly-detection results depend on the training data and selected threshold.
7. **No verified physical intervention:** The current prototype does not control a motor or validate corrective actions on physical equipment.
8. **No guaranteed safety decision:** Model outputs should not be used as the sole basis for industrial safety or maintenance decisions.

---

## 18. Future Improvements

Potential extensions include:

### Real-time sensor integration

Connect the application to a sensor interface, MQTT broker, industrial gateway, or another supported data source.

### Future-temperature forecasting

Train and validate a forecasting model that uses historical sequences to predict motor temperature at a future time horizon.

### Decision-support engine

Add a module to compare simulated operating scenarios, estimate their expected outcomes, and provide recommendations with clear uncertainty and assumptions.

Any scenario-based results should be labeled as simulated estimates rather than verified physical outcomes.

### Improved fault analysis

Evaluate supervised classification if reliable fault labels are available. Compare model predictions against independent test runs and report precision, recall, F1 score, and false-alarm rates.

### Better validation

Use separate runs for training and testing, investigate data leakage, and evaluate model stability across operating conditions.

### Dashboard improvements

Possible additions include:

- Historical event logs.
- Configurable anomaly thresholds.
- Exportable reports.
- Dataset selection and validation.
- Model-performance summaries.
- Clear distinction between measured and predicted values.

### Deployment

Package the application for a suitable hosting environment, add environment-specific configuration, and document how model artifacts and datasets are supplied.

---

## 19. Contributing

Contributions from teammates are welcome.

### Suggested workflow

1. Clone the repository.
2. Create a separate branch for your change.
3. Make and test the changes locally.
4. Confirm that the dashboard starts successfully.
5. Check that model input columns and saved artifacts remain compatible.
6. Commit your changes.
7. Push the branch and open a pull request.

Example:

```bash
git checkout -b feature/your-feature
```

After making and testing your changes:

```bash
git add .
git commit -m "Describe your change"
git push -u origin feature/your-feature
```

Keep commits focused, document changes to the dataset or model features, and avoid committing secrets, local virtual environments, or unrelated generated files.

If a model artifact is regenerated, document the training-data and evaluation changes so teammates can understand why the results may differ.

---

## 20. Disclaimer

Auto Twin is an educational and experimental software prototype for motor monitoring and digital-twin research.

Its anomaly flags and temperature estimates are model outputs and may be incorrect. They are not certified fault diagnoses, validated predictions of physical equipment behavior, or guarantees of motor safety.

Do not connect the prototype to safety-critical control systems or use its outputs as the sole basis for operating, stopping, or maintaining industrial equipment without appropriate independent engineering validation.

---

## Maintainers

**Repository:** [Raghavuchiha/digital-twin-motor](https://github.com/Raghavuchiha/digital-twin-motor)

For questions, improvements, or bug reports, open an issue in the repository or discuss the change with the project team.
