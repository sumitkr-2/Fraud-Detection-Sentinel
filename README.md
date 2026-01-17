
# 🛡️ Fraud Detection Sentinel
### *AI-Powered Anomaly Detection System for Enrolment Fraud*

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![ML](https://img.shields.io/badge/AI-IsolationForest-green)

---

## 📌 Project Overview
**Fraud Detection Sentinel** is a comprehensive analytics dashboard designed to identify suspicious patterns in biometric and demographic enrolment data.

While traditional systems rely on simple thresholds, this project uses **Unsupervised Machine Learning (Isolation Forest)** to detect multi-dimensional anomalies. It analyzes the relationship between adult enrolments, demographic updates, and biometric changes to flag high-risk pincodes that deviate from the norm.

---

## 🚀 Key Features
- **Interactive Dashboard:** A user-friendly web interface built with **Streamlit** that allows real-time data filtering and analysis.
- **3D Anomaly Visualization:** Interactive 3D scatter plots (via **Plotly**) to visualize complex relationships between three risk factors simultaneously.
- **Explainable AI:** Automated risk descriptions explaining *why* a specific location was flagged (e.g., *“Critical spike in Biometric Updates”*).
- **Dual Operation Mode:**
  - **Upload Mode:** Analyze your own CSV data.
  - **Demo Mode:** Use built-in synthetic data for instant demonstration.
- **Geospatial Insights:** District-wise breakdown of fraud hotspots.

---

## 🛠️ Tech Stack
- **Language:** Python 3.10
- **Frontend:** Streamlit
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn (Isolation Forest)
- **Visualization:** Plotly Express, Matplotlib / Seaborn (research notebook)

---

## 📂 Project Structure

```text
fraud-sentinel/
├── app.py              # Main Streamlit dashboard
├── server.ipynb        # Model research & data exploration
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── data/               # (Optional) Input CSV files
````

---

## ⚙️ Installation & Setup

### 1. Prerequisites

Ensure **Python 3.10** is installed.

---

### 2. Clone the Repository

```bash
git clone https://github.com/sumitkr-2/Fraud-Detection-Sentinel.git
cd fraud-sentinel
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ How to Run

### Run the Web Dashboard (Recommended)

```bash
streamlit run app.py
```

The app will open at:
👉 `http://localhost:8501`

---

## 📸 Screenshots

<img width="1470" height="830" alt="image" src="https://github.com/user-attachments/assets/78201064-cd65-4f80-a35a-e77db6fa75e0" />
<img width="1099" height="642" alt="image" src="https://github.com/user-attachments/assets/0d008487-5022-496b-ac06-e81e70a2f02a" />
<img width="1080" height="515" alt="image" src="https://github.com/user-attachments/assets/975cc286-4f0a-47e4-bbf9-e4ff084c3cc9" />



---

## 🧠 How the Model Works

The system aggregates data by **Pincode** and feeds three key features into an **Isolation Forest** model:

1. **`age_18_greater`** – New adult enrolments (possible ghost beneficiaries)
2. **`demo_age_17_`** – Demographic update frequency
3. **`total_bio_updates`** – Biometric update volume

Data points isolated in feature space are flagged as **anomalies** (`Score = -1`).

---

## 📊 Sample Data Format

Uploaded CSV files must follow these schemas:

### Enrolment CSV

* `state`
* `district`
* `pincode`
* `age_18_greater`

### Demographic CSV

* `state`
* `district`
* `pincode`
* `demo_age_17_`

### Biometric CSV

* `state`
* `district`
* `pincode`
* `bio_age_5_17`
* `bio_age_17_`

---

## ✅ Use Cases & Impact

* Early detection of enrolment fraud and ghost beneficiaries
* Identification of abnormal biometric update patterns
* District-level risk monitoring for targeted audits
* Data-driven decision support for government agencies

---

## 📜 License

© 2026 Sumit Kumar

This project is created and owned by me.  
It is shared here for learning, demonstration, and evaluation purposes.

Please do not copy or reuse this project as your own work without permission.

