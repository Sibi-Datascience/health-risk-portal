# 🏥 Enterprise Population Health & Risk Stratification Portal

An end-to-end machine learning system designed to predict 30-day hospital readmissions using real clinical data. This portal processes patient metrics in real-time to stratify profiles into operational risk tiers, allowing healthcare networks to distribute medical resources and preventative care efficiently.

🔗 **Live Production App:** [https://huggingface.co/spaces/Sibikrish03/health-risk-portal](https://huggingface.co/spaces/Sibikrish03/health-risk-portal)

---

## 🚀 Key Features & Architectural Fixes

* **Patient-Grouped Validation (Leak-Proof):** Implements a robust `GroupShuffleSplit` strategy ensuring a patient's historical records are isolated entirely to either the train or test partition. This eliminates data leakage caused by multi-visit patients spanning across splits.
* **Target Leakage Safeguards:** Integrates an explicit post-encoding runtime assert block validating that the high-risk target cannot silently masquerade inside the training matrix.
* **Clinical Eligibility Filtering:** Programmatically removes expired or hospice-discharged encounters (`discharge_disposition_id` codes 11, 13, 14, 19, 20, 21) prior to feature pipeline ingestion. Patients who pass away or transition to terminal care are logically ineligible for administrative readmission, making their removal necessary for unbiased training.
* **State Caching Optimization:** Leverages Streamlit state decorators (`@st.cache_resource`) to lock the pre-trained binaries continuously in shared server memory, dropping look-up response latency during concurrent user interaction.

---

## 🛠️ The Tech Stack

* **Data Engineering & Modeling:** Python, Scikit-Learn, Pandas, NumPy
* **Explainability Engine:** SHAP (SHapley Additive exPlanations) for live, per-patient feature contributions.
* **Serialization Protocol:** Joblib (Binary state extraction)
* **Frontend Interface & Hosting:** Streamlit UI Architecture deployed inside a public cloud container on Hugging Face Spaces.

---

## 📊 Clinical Benchmarks & Performance Edge

Every figure below comes directly from this project's own patient-grouped validation run—comparing the production Random Forest ensemble against a Logistic Regression linear baseline trained and evaluated on an identical split of **99,343 real hospital encounters**. 

Because readmission events exhibit severe class imbalance (~11.39% prevalence), the model is deliberately tuned toward high-risk recall to prioritize catching critical events over administrative false alarms.

### Head-to-Head Validation Metrics

| Metric | Production RF Ensemble | LR Linear Baseline | Clinical / Business Impact |
| :--- | :--- | :--- | :--- |
| **ROC-AUC (Test Set)** | **0.644** | 0.637 | Higher AUC means the ensemble ranks at-risk patients above stable ones more reliably across every threshold. |
| **5-Fold CV ROC-AUC** | **0.647 (± 0.005)** | *Not separately tracked* | Confirms the ensemble's score is stable across patient-grouped folds. |
| **High-Risk Recall** | **54%** | *Not separately tracked* | Catching high-risk patients is the primary clinical goal — missing one costs far more than a false alarm. |
| **High-Risk Precision** | **17%** | *Not separately tracked* | Deliberate trade-off: the model is tuned toward recall, so most flags are followed up even if not all are truly high-risk. |
| **High-Risk F1** | **0.259** | *Not separately tracked* | Balanced precision/recall score for the minority class. |
| **Stable-Class Recall** | **66%** | *Not separately tracked* | Of truly stable patients, this share is correctly identified and spared unnecessary intervention. |

> **Note:** Logistic Regression entries marked *"not separately tracked"* reflect that the validation run measured baseline AUC for both models but ran the full classification report exclusively on the production model.

### Production Random Forest Confusion Matrix ($n = 19,802$ Test Set)

| Actual / Predicted | Predicted: Stable | Predicted: High Risk |
| :--- | :--- | :--- |
| **Actual: Stable** | **11,582** *(True Negatives)* | **6,004** *(False Positives)* |
| **Actual: High Risk** | **1,015** *(False Negatives)* | **1,201** *(True Positives)* |

---

## 🧬 Feature Engineering & Key Predictive Drivers

The dataset features were enriched with domain-specific interactions including a calculated `clinical_severity_index` (time_in_hospital × num_lab_procedures) and a `total_complexity_score`. The top feature coefficients impacting final risk probability scores consist of:

1. 📈 **number_inpatient** (Prior 12-month admission history) — **14.90% importance**
2. 🔄 **discharge_disposition_id** (Discharge tracking status) — **5.47% importance**
3. 🚨 **number_emergency** (Prior emergency visits) — **4.33% importance**
4. 🧮 **total_complexity_score** (Procedures + Medications + Emergency) — **3.63% importance**
