# 🏥 Enterprise Population Health & Risk Stratification Portal

An end-to-end machine learning system designed to predict 30-day hospital readmissions using real clinical data. This portal processes patient metrics in real-time to stratify profiles into operational risk tiers, allowing healthcare networks to distribute medical resources and preventative care efficiently.

🔗 **Live Production App:** [https://huggingface.co/spaces/Sibikrish03/health-risk-portal]

---

## 🚀 Key Features & Architectural Fixes
* **Patient-Grouped Validation (Leak-Proof):** Implements a robust `GroupShuffleSplit` strategy ensuring a patient's historical records are isolated entirely to either the train or test partition. This eliminates data leakage caused by multi-visit patients spanning across splits.
* **Target Leakage Safeguards:** Integrates an explicit post-encoding runtime `assert` block validating that the high-risk target cannot silently masquerade inside the training matrix.
* **Clinical Eligibility Filtering:** Programmatically removes expired or hospice-discharged encounters (`discharge_disposition_id` codes 11, 13, 14, 19, 20, 21) prior to feature pipeline ingestion. Patients who pass away or transition to terminal care are logically ineligible for administrative readmission, making their removal necessary for unbiased training.
* **State Caching Optimization:** Leverages Streamlit state decorators (`@st.cache_resource`) to lock the pre-trained binaries continuously in shared server memory, dropping look-up response latency during concurrent user interaction.

## 🛠️ The Tech Stack
* **Data Engineering & Modeling:** Python, Scikit-Learn, Pandas, NumPy
* **Serialization Protocol:** Joblib (Binary state extraction)
* **Frontend Interface & Hosting:** Streamlit UI Architecture deployed inside a public cloud container on Hugging Face Spaces.

## 📊 Core Evaluated Metrics
The dataset includes **99,343 clinical records** after rigorous demographic cleaning. Because readmission events exhibit severe class imbalance (~11.39% prevalence), the optimization pipeline evaluates performance across multiple structural dimensions beyond basic accuracy:

### Model Performance (Leak-Proof Patient-Grouped Evaluation)
| Evaluation Metric | Baseline Logistic Regression | Production Random Forest Classifier |
| :--- | :---: | :---: |
| **Area Under ROC (ROC-AUC)** | 0.6370 | **0.6442** |
| **High-Risk Class Recall (Sensitivity)** | 0.54 | **0.54** |
| **High-Risk Class Precision** | 0.16 | **0.17** |

### Production Random Forest Confusion Matrix
* **True Negatives** (Correctly predicted stable): **11,582**
* **True Positives** (Accurately identified high-risk events): **1,201**
* **False Negatives** (Missed high-risk patients): **1,015**
* **False Positives** (Administrative false alarms): **6,004**

### Feature Engineering & Key Predictive Drivers
The dataset features were enriched with domain-specific interactions including a calculated `clinical_severity_index` (`time_in_hospital` $\times$ `num_lab_procedures`) and a `total_complexity_score`. The top feature coefficients impacting final risk probability scores consist of:
1. `number_inpatient` (Prior 12-month admission history) — **14.90% importance**
2. `discharge_disposition_id` (Discharge tracking status) — **5.47% importance**
3. `number_emergency` (Prior emergency visits) — **4.33% importance**
4. `total_complexity_score` (Procedures + Medications + Emergency) — **3.63% importance**
