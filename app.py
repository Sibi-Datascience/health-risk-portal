import time
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib import gridspec

st.set_page_config(page_title="Readmission Risk Intelligence Platform", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    .stApp { background-color: #0b0f19; }
    .block-container { padding-top: 1.6rem; max-width: 1300px; }
    h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 700; color: #f1f5f9; }
    .platform-title { font-size: 2rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.02em; margin-bottom: 0.1rem; }
    .platform-subtitle { color: #8b95a7; font-size: 0.95rem; margin-bottom: 1.2rem; }
    .status-pill {
        display: inline-block; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem;
        letter-spacing: 0.06em; padding: 3px 10px; border-radius: 20px; margin-left: 10px;
        background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.35);
        vertical-align: middle;
    }
    div[data-testid="stExpander"] {
        background-color: #111827; border: 1px solid #1f2937; border-radius: 10px;
    }
    div[data-testid="stForm"] { border: none; }
    .stButton button, .stFormSubmitButton button {
        background-color: #111827; color: #e5e7eb; border: 1px solid #1f2937;
        border-radius: 8px; font-weight: 600; font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stButton button:hover { border-color: #06b6d4; color: #06b6d4; }
    .stFormSubmitButton button {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: #04141a; border: none; font-weight: 700;
    }
    div[data-testid="stMetricValue"] { font-size: 1.7rem; color: #f1f5f9; font-family: 'JetBrains Mono', monospace; }
    div[data-testid="stMetricLabel"] { color: #8b95a7; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid #1f2937; }
    .stTabs [data-baseweb="tab"] { color: #8b95a7; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #06b6d4 !important; }
    .disclaimer {
        font-size: 0.82rem; color: #6b7280; border-top: 1px solid #1f2937;
        padding-top: 0.9rem; margin-top: 2rem;
    }
    html, body { overflow: visible !important; height: auto !important; }
    [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {
        overflow: visible !important; height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

RF_METRICS = {
    "test_auc": 0.644, "cv_auc_mean": 0.647, "cv_auc_std": 0.005,
    "precision_high_risk": 0.17, "recall_high_risk": 0.54,
    "precision_stable": 0.92, "recall_stable": 0.66,
    "tn": 11582, "fp": 6004, "fn": 1015, "tp": 1201, "n_test": 19802,
}
RF_METRICS["f1_high_risk"] = round(
    2 * RF_METRICS["precision_high_risk"] * RF_METRICS["recall_high_risk"]
    / (RF_METRICS["precision_high_risk"] + RF_METRICS["recall_high_risk"]), 3
)
LR_METRICS = {"test_auc": 0.637}

TIER_CRITICAL = 0.54
TIER_MODERATE = 0.48

MEDICATIONS = ['metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
               'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
               'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide',
               'insulin','glyburide-metformin','glipizide-metformin',
               'glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone']

PRIMARY_DIAGNOSES = {
    "Other / unspecified": None,
    "Diabetes, uncomplicated (250)": "250",
    "Diabetes with complications (250.01)": "250.01",
    "Congestive heart failure (428)": "428",
    "Ischemic heart disease (414)": "414",
    "Acute myocardial infarction (410)": "410",
    "Cardiac dysrhythmias (427)": "427",
    "Essential hypertension (401)": "401",
    "Pneumonia (486)": "486",
    "Chronic bronchitis / COPD (491)": "491",
    "Osteoarthritis (715)": "715",
    "Urinary / renal disorder (599)": "599",
}

ADMISSION_TYPES = {"Emergency":1,"Urgent":2,"Elective":3,"Newborn":4,"Trauma center":7,"Not available":5}
ADMISSION_SOURCES = {"Emergency room":7,"Physician referral":1,"Clinic referral":2,
                      "Transfer from a hospital":4,"Transfer from skilled nursing facility":5,
                      "Transfer from another health care facility":6}
DISCHARGE_DISPOSITIONS = {"Discharged to home":1,"Transferred to another short-term hospital":2,
                           "Transferred to skilled nursing facility":3,
                           "Discharged home with home health service":6,
                           "Left against medical advice":7,
                           "Transferred to rehabilitation facility":22,
                           "Transferred to long-term care hospital":23}

NUMERIC_SLIDER_RANGES = {
    'time_in_hospital':(1,14),'num_lab_procedures':(1,130),'num_procedures':(0,6),
    'num_medications':(1,80),'number_outpatient':(0,40),'number_emergency':(0,60),
    'number_inpatient':(0,20),'number_diagnoses':(1,16),
}

_BASELINE_MEDS = {f"med_{d}": "No (not prescribed)" for d in MEDICATIONS}

HIGH_RISK_EXAMPLE = {
    'race_input':'Caucasian','gender_input':'Male','age_input':'[70-80)',
    'adm_type_input':'Emergency','adm_source_input':'Emergency room',
    'discharge_input':'Transferred to skilled nursing facility',
    'time_in_hospital_input':13,'num_lab_procedures_input':95,'num_procedures_input':3,
    'num_medications_input':32,'number_outpatient_input':3,'number_emergency_input':4,
    'number_inpatient_input':8,'number_diagnoses_input':16,
    'diag_input':'Congestive heart failure (428)','glu_input':'>300','a1c_input':'>8',
    'specialty_input':'InternalMedicine','change_input':'Regimen changed','diabmed_input':'Yes',
    **{**_BASELINE_MEDS,'med_metformin':'Up (dose increased)','med_insulin':'Up (dose increased)',
       'med_glyburide':'Up (dose increased)'},
}

STABLE_EXAMPLE = {
    'race_input':'Caucasian','gender_input':'Female','age_input':'[40-50)',
    'adm_type_input':'Elective','adm_source_input':'Physician referral',
    'discharge_input':'Discharged to home',
    'time_in_hospital_input':2,'num_lab_procedures_input':18,'num_procedures_input':0,
    'num_medications_input':8,'number_outpatient_input':0,'number_emergency_input':0,
    'number_inpatient_input':0,'number_diagnoses_input':3,
    'diag_input':'Diabetes, uncomplicated (250)','glu_input':'Normal','a1c_input':'Normal',
    'specialty_input':'Family/GeneralPractice','change_input':'No change','diabmed_input':'Yes',
    **{**_BASELINE_MEDS,'med_metformin':'Steady (no change)','med_insulin':'Steady (no change)'},
}

def load_example(example):
    for key, value in example.items():
        st.session_state[key] = value

FRIENDLY_NUMERIC = {
    'time_in_hospital':'Length of hospital stay (days)',
    'num_lab_procedures':'Number of lab procedures',
    'num_procedures':'Number of procedures',
    'num_medications':'Number of medications administered',
    'number_outpatient':'Prior outpatient visits (past year)',
    'number_emergency':'Prior emergency visits (past year)',
    'number_inpatient':'Prior inpatient visits (past year)',
    'number_diagnoses':'Number of diagnoses recorded',
    'med_change_count':'Medication dose changes during stay',
    'clinical_severity_index':'Clinical severity index (stay length x lab volume)',
    'total_complexity_score':'Care complexity score',
    'admission_type_id':'Admission type',
    'discharge_disposition_id':'Discharge disposition',
    'admission_source_id':'Admission source',
}

@st.cache_resource
def load_artifacts():
    model = joblib.load('optimized_random_forest_model.pkl')
    schema = joblib.load('production_feature_schema.joblib')
    return model, schema

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

def style_dark_axes(fig, ax):
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')
    ax.tick_params(colors='#9ca3af', labelsize=9)
    ax.xaxis.label.set_color('#9ca3af')
    ax.yaxis.label.set_color('#9ca3af')
    ax.title.set_color('#e5e7eb')
    for spine in ('bottom','left'):
        ax.spines[spine].set_color('#1f2937')
    for spine in ('top','right'):
        ax.spines[spine].set_visible(False)

@st.cache_resource
def build_importance_chart(_model, _schema):
    importances = _model.feature_importances_
    imp_df = pd.DataFrame({'Feature':_schema,'Importance':importances})
    imp_df['Label'] = imp_df['Feature'].apply(friendly_name)
    imp_df = imp_df.sort_values('Importance',ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.barh(imp_df['Label'], imp_df['Importance'], color='#06b6d4')
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 drivers of readmission risk across all patients", fontsize=12, pad=10)
    style_dark_axes(fig, ax)
    fig.tight_layout()
    return fig

try:
    model, schema = load_artifacts()
except Exception as e:
    st.error(
        "The trained model could not be loaded. This app requires "
        "'optimized_random_forest_model.pkl' and 'production_feature_schema.joblib' "
        "to be present in the application directory. No fallback prediction logic "
        "is used -- if the model is unavailable, the app stops here."
    )
    st.exception(e)
    st.stop()

explainer = get_explainer(model)

def drug_options(drug):
    statuses_present = sorted([c[len(drug)+1:] for c in schema if c.startswith(drug+'_')])
    if statuses_present == ['No','Steady','Up']:
        return {'Down (dose decreased)':None,'No (not prescribed)':f'{drug}_No',
                'Steady (no change)':f'{drug}_Steady','Up (dose increased)':f'{drug}_Up'}
    elif statuses_present == ['Steady','Up']:
        return {'No (not prescribed)':None,'Steady (no change)':f'{drug}_Steady',
                'Up (dose increased)':f'{drug}_Up'}
    elif statuses_present == ['Steady']:
        return {'No (not prescribed)':None,'Steady (no change)':f'{drug}_Steady'}
    else:
        opts = {'Baseline':None}
        for s in statuses_present: opts[s] = f'{drug}_{s}'
        return opts

def friendly_name(col):
    if col in FRIENDLY_NUMERIC: return FRIENDLY_NUMERIC[col]
    prefixes = {
        'race_':'Race','gender_':'Gender','age_':'Age group',
        'medical_specialty_':'Admitting specialty','max_glu_serum_':'Glucose serum test',
        'A1Cresult_':'A1C test result','diag_1_':'Primary diagnosis code',
    }
    for prefix, label in prefixes.items():
        if col.startswith(prefix):
            return f"{label}: {col[len(prefix):]}"
    if col == 'change_No': return "No change to medication regimen"
    if col == 'diabetesMed_Yes': return "Prescribed diabetes medication"
    if '_' in col:
        drug, status = col.rsplit('_',1)
        return f"{drug.replace('-',' / ').title()}: {status}"
    return col

def assign_tier(prob):
    if prob >= TIER_CRITICAL: return 'Critical'
    elif prob >= TIER_MODERATE: return 'Moderate'
    return 'Stable'

TIER_STYLE = {
    'Critical': dict(accent='#ef4444', glow='rgba(239,68,68,0.45)', bg='rgba(239,68,68,0.08)',
                     label='CRITICAL RISK PROFILE \u2014 IMMEDIATE CARE MANAGEMENT AUDIT',
                     action='Route directly to care management for intensive, individualized nurse-led intervention before discharge.'),
    'Moderate': dict(accent='#f59e0b', glow='rgba(245,158,11,0.40)', bg='rgba(245,158,11,0.08)',
                     label='MODERATE RISK \u2014 PROACTIVE TELEHEALTH MONITORING',
                     action='Enroll in proactive telehealth monitoring with scheduled follow-up calls during the highest-risk window after discharge.'),
    'Stable':   dict(accent='#10b981', glow='rgba(16,185,129,0.40)', bg='rgba(16,185,129,0.08)',
                     label='HEALTH COMPLIANT \u2014 ROUTINE FOLLOW-UP',
                     action='Assign to standard preventative care and routine follow-up scheduling.'),
}

def build_feature_vector(values):
    vec = pd.DataFrame(np.zeros((1,len(schema))), columns=schema)
    vec.loc[0,'admission_type_id']      = values['admission_type_id']
    vec.loc[0,'discharge_disposition_id']= values['discharge_disposition_id']
    vec.loc[0,'admission_source_id']    = values['admission_source_id']
    vec.loc[0,'time_in_hospital']       = values['time_in_hospital']
    vec.loc[0,'num_lab_procedures']     = values['num_lab_procedures']
    vec.loc[0,'num_procedures']         = values['num_procedures']
    vec.loc[0,'num_medications']        = values['num_medications']
    vec.loc[0,'number_outpatient']      = values['number_outpatient']
    vec.loc[0,'number_emergency']       = values['number_emergency']
    vec.loc[0,'number_inpatient']       = values['number_inpatient']
    vec.loc[0,'number_diagnoses']       = values['number_diagnoses']
    med_cols = [c for c in values['med_cols'] if c is not None]
    vec.loc[0,'med_change_count']         = sum(1 for c in med_cols if c.endswith('_Up') or c.endswith('_Down'))
    vec.loc[0,'clinical_severity_index']  = values['time_in_hospital'] * values['num_lab_procedures']
    vec.loc[0,'total_complexity_score']   = values['num_procedures'] + values['num_medications'] + values['number_emergency']
    for col in med_cols:
        vec.loc[0, col] = 1
    for col in [values['race_col'],values['gender_col'],values['age_col'],
                values['specialty_col'],values['diag1_col'],values['glu_col'],
                values['a1c_col'],values['change_col'],values['diabmed_col']]:
        if col is not None:
            vec.loc[0, col] = 1
    return vec

def render_verdict_panel(prob, tier, pred_ms, shap_ms):
    s = TIER_STYLE[tier]
    pct = prob * 100
    st.markdown(f"""
    <div style="background:{s['bg']};border:1px solid {s['accent']}40;border-left:4px solid {s['accent']};
                border-radius:12px;padding:22px 26px;margin-bottom:0.8rem;
                box-shadow:0 0 28px {s['glow']};">
      <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;letter-spacing:1.5px;
                   color:{s['accent']};font-weight:600;">{s['label']}</div>
      <div style="display:flex;align-items:baseline;gap:10px;margin-top:10px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:42px;font-weight:700;color:{s['accent']};">{pct:.1f}%</span>
        <span style="font-size:13.5px;color:#9ca3af;">posterior 30-day readmission probability</span>
      </div>
      <div style="background:#1f2937;border-radius:6px;height:10px;margin-top:14px;overflow:hidden;">
        <div style="background:linear-gradient(90deg,{s['accent']}aa,{s['accent']});width:{min(pct,100):.1f}%;
                    height:100%;border-radius:6px;box-shadow:0 0 12px {s['glow']};"></div>
      </div>
      <p style="margin-top:14px;color:#d1d5db;font-size:14px;line-height:1.5;">{s['action']}</p>
      <div style="margin-top:16px;padding-top:12px;border-top:1px solid #1f293780;
                   display:flex;gap:28px;font-family:'JetBrains Mono',monospace;font-size:11px;color:#6b7280;">
        <span>INFERENCE &nbsp;<span style="color:#9ca3af;">{pred_ms:.2f} ms</span></span>
        <span>EXPLANATION &nbsp;<span style="color:#9ca3af;">{shap_ms:.1f} ms</span></span>
        <span>THRESHOLDS &nbsp;<span style="color:#9ca3af;">{TIER_MODERATE*100:.0f}% / {TIER_CRITICAL*100:.0f}%</span></span>
      </div>
    </div>""", unsafe_allow_html=True)

def extract_shap_vector(raw_shap, n):
    arr = raw_shap
    if isinstance(arr, list):
        arr = arr[1] if len(arr) > 1 else arr[0]
        return np.asarray(arr)[0]
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return arr[0,:,1] if arr.shape[-1] >= 2 else arr[0,:,0]
    if arr.ndim == 2:
        return arr[0] if arr.shape[-1] == n else arr.flatten()[:n]
    return arr.flatten()[:n]

def compute_shap_explanation(vec):
    raw_shap = explainer.shap_values(vec)
    final_shap = extract_shap_vector(raw_shap, len(schema))
    input_row = vec.iloc[0]
    candidate_idx = [i for i,col in enumerate(schema) if col in FRIENDLY_NUMERIC or input_row[col]==1]
    candidates = [(schema[i], final_shap[i], input_row[schema[i]]) for i in candidate_idx]
    candidates.sort(key=lambda x: abs(x[1]), reverse=True)
    top = candidates[:8][::-1]
    labels, vals, colors, raw_vals = [], [], [], []
    for name, val, raw_val in top:
        label = friendly_name(name)
        if name in FRIENDLY_NUMERIC: label = f"{label} = {raw_val:g}"
        labels.append(label); vals.append(val); raw_vals.append((name, raw_val))
        colors.append('#ef4444' if val > 0 else '#10b981')
    return labels, vals, colors, raw_vals

def render_explanation_panel(labels, vals, colors, raw_vals):
    fig = plt.figure(figsize=(13, 4.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.32)
    ax1 = fig.add_subplot(gs[0])
    ax1.barh(labels, vals, color=colors)
    ax1.axvline(0, color='#4b5563', linewidth=0.8)
    ax1.set_xlabel("Contribution to predicted risk (SHAP value)")
    ax1.set_title("Why this patient received this score", fontsize=12, pad=10)
    style_dark_axes(fig, ax1)

    ax2 = fig.add_subplot(gs[1])
    numeric_pts = [(friendly_name(n).split(' = ')[0], v, NUMERIC_SLIDER_RANGES[n])
                    for n,v in raw_vals if n in NUMERIC_SLIDER_RANGES]
    if numeric_pts:
        names = [p[0] for p in numeric_pts][::-1]
        fracs = [min(max((p[1]-p[2][0])/max(p[2][1]-p[2][0],1e-9),0),1) for p in numeric_pts][::-1]
        bar_colors = ['#06b6d4' if f < 0.7 else '#ef4444' for f in fracs]
        ax2.barh(names, fracs, color=bar_colors)
        ax2.set_xlim(0,1)
        ax2.set_xlabel("Position within valid input range")
        ax2.set_title("Patient value vs. plausible range", fontsize=12, pad=10)
    else:
        ax2.text(0.5,0.5,"No numeric drivers\nin top factors",ha='center',va='center',
                  color='#9ca3af',fontsize=10,transform=ax2.transAxes)
        ax2.set_xticks([]); ax2.set_yticks([])
    style_dark_axes(fig, ax2)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Left: red bars push predicted risk higher, green bars pull it lower (per-patient SHAP attribution). "
        "Right: where this patient's numeric inputs fall within their slider's valid range."
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="platform-title">Readmission Risk Intelligence Platform'
    '<span class="status-pill">MODEL ONLINE</span></div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="platform-subtitle">Predicts 30-day diabetic readmission risk from a Random Forest '
    'ensemble trained on 99,343 real hospital encounters, with live per-patient SHAP explainability.</div>',
    unsafe_allow_html=True
)

tab_assess, tab_bench, tab_about = st.tabs(
    ["Risk Assessment", "Clinical Benchmarks & Performance Edge", "About This Model"]
)

# ---------------------------------------------------------------------------
# TAB 1
# ---------------------------------------------------------------------------
with tab_assess:
    st.subheader("Patient Information")
    st.caption("Inject a demo profile to see the model in action, or fill in the form yourself.")
    d1, d2, _ = st.columns([1,1,2])
    d1.button("Inject High-Risk Profile", on_click=load_example, args=(HIGH_RISK_EXAMPLE,))
    d2.button("Inject Stable Baseline Profile", on_click=load_example, args=(STABLE_EXAMPLE,))

    with st.form("patient_form"):
        with st.expander("Demographics", expanded=True):
            c1,c2,c3 = st.columns(3)
            race_options = {"African American":None,"Caucasian":"race_Caucasian","Asian":"race_Asian",
                             "Hispanic":"race_Hispanic","Other":"race_Other","Unknown":"race_Unknown"}
            race_label = c1.selectbox("Race", list(race_options.keys()), index=1, key="race_input")
            gender_options = {"Female":None,"Male":"gender_Male","Unknown/Invalid":"gender_Unknown/Invalid"}
            gender_label = c2.selectbox("Gender", list(gender_options.keys()), index=0, key="gender_input")
            age_brackets = ["[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
                             "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)"]
            age_label = c3.selectbox("Age group", age_brackets, index=6, key="age_input")

        with st.expander("Admission Details", expanded=True):
            c1,c2,c3 = st.columns(3)
            adm_type_label    = c1.selectbox("Admission type", list(ADMISSION_TYPES.keys()), index=0, key="adm_type_input")
            adm_source_label  = c2.selectbox("Admission source", list(ADMISSION_SOURCES.keys()), index=0, key="adm_source_input")
            discharge_label   = c3.selectbox("Discharge disposition", list(DISCHARGE_DISPOSITIONS.keys()), index=0, key="discharge_input")

        with st.expander("Clinical Metrics", expanded=True):
            c1,c2,c3,c4 = st.columns(4)
            time_in_hospital    = c1.slider("Length of stay (days)", 1, 14, 4,  key="time_in_hospital_input")
            num_lab_procedures  = c2.slider("Lab procedures",         1, 130, 43, key="num_lab_procedures_input")
            num_procedures      = c3.slider("Procedures performed",   0, 6, 1,   key="num_procedures_input")
            num_medications     = c4.slider("Medications administered",1, 80, 15, key="num_medications_input")
            c5,c6,c7,c8 = st.columns(4)
            number_outpatient  = c5.slider("Prior outpatient visits", 0, 40, 0,  key="number_outpatient_input")
            number_emergency   = c6.slider("Prior emergency visits",  0, 60, 0,  key="number_emergency_input")
            number_inpatient   = c7.slider("Prior inpatient visits",  0, 20, 0,  key="number_inpatient_input")
            number_diagnoses   = c8.slider("Number of diagnoses",     1, 16, 7,  key="number_diagnoses_input")

        with st.expander("Diagnosis and Lab Results"):
            c1,c2,c3 = st.columns(3)
            diag_label = c1.selectbox("Primary diagnosis", list(PRIMARY_DIAGNOSES.keys()), index=0, key="diag_input")
            glu_options = {">200":None,">300":"max_glu_serum_>300","Normal":"max_glu_serum_Norm",
                           "Not measured":"max_glu_serum_Not Measured"}
            glu_label = c2.selectbox("Glucose serum test result", list(glu_options.keys()), index=3, key="glu_input")
            a1c_options = {">7":None,">8":"A1Cresult_>8","Normal":"A1Cresult_Norm",
                           "Not measured":"A1Cresult_Not Measured"}
            a1c_label = c3.selectbox("A1C test result", list(a1c_options.keys()), index=3, key="a1c_input")

        with st.expander("Medication Regimen (21 tracked medications)"):
            c1,c2 = st.columns(2)
            specialty_options = sorted(c[len('medical_specialty_'):] for c in schema
                                        if c.startswith('medical_specialty_'))
            specialty_options = ["Other / not listed"] + specialty_options
            default_sp = specialty_options.index('InternalMedicine') if 'InternalMedicine' in specialty_options else 0
            specialty_label = c1.selectbox("Admitting specialty", specialty_options, index=default_sp, key="specialty_input")
            change_options  = {"Regimen changed":None,"No change":"change_No"}
            change_label    = c2.selectbox("Medication regimen changed during stay", list(change_options.keys()), index=1, key="change_input")
            diabmed_options = {"Yes":"diabetesMed_Yes","No":None}
            diabmed_label   = c2.selectbox("Currently prescribed diabetes medication", list(diabmed_options.keys()), index=0, key="diabmed_input")

            st.markdown("**Individual medication status**")
            med_choices = {}
            cols = st.columns(3)
            for i, drug in enumerate(MEDICATIONS):
                opts = drug_options(drug)
                default_idx = 0
                if drug in ('insulin','metformin') and 'Steady (no change)' in opts:
                    default_idx = list(opts.keys()).index('Steady (no change)')
                choice_label = cols[i%3].selectbox(drug.replace('-',' / ').title(), list(opts.keys()),
                                                    index=default_idx, key=f"med_{drug}")
                med_choices[drug] = opts[choice_label]

        st.markdown("---")
        run_prediction = st.form_submit_button("Run Diagnostic Evaluation")

    result_col, chart_col = st.columns([1,1.5])

    if run_prediction:
        values = {
            'admission_type_id':       ADMISSION_TYPES[adm_type_label],
            'discharge_disposition_id':DISCHARGE_DISPOSITIONS[discharge_label],
            'admission_source_id':     ADMISSION_SOURCES[adm_source_label],
            'time_in_hospital':        time_in_hospital,
            'num_lab_procedures':      num_lab_procedures,
            'num_procedures':          num_procedures,
            'num_medications':         num_medications,
            'number_outpatient':       number_outpatient,
            'number_emergency':        number_emergency,
            'number_inpatient':        number_inpatient,
            'number_diagnoses':        number_diagnoses,
            'med_cols':                list(med_choices.values()),
            'race_col':                race_options[race_label],
            'gender_col':              gender_options[gender_label],
            'age_col':                 f"age_{age_label}",
            'specialty_col':           None if specialty_label=="Other / not listed" else f"medical_specialty_{specialty_label}",
            'diag1_col':               (f"diag_1_{PRIMARY_DIAGNOSES[diag_label]}" if PRIMARY_DIAGNOSES[diag_label] else None),
            'glu_col':                 glu_options[glu_label],
            'a1c_col':                 a1c_options[a1c_label],
            'change_col':              change_options[change_label],
            'diabmed_col':             diabmed_options[diabmed_label],
        }
        with st.spinner("Running inference and computing explanation..."):
            vec = build_feature_vector(values)
            t0 = time.perf_counter()
            probability = model.predict_proba(vec)[0,1]
            pred_ms = (time.perf_counter()-t0)*1000
            t1 = time.perf_counter()
            shap_labels, shap_vals, shap_colors, shap_raw = compute_shap_explanation(vec)
            shap_ms = (time.perf_counter()-t1)*1000
            tier = assign_tier(probability)

        with result_col:
            render_verdict_panel(probability, tier, pred_ms, shap_ms)
            st.caption(
                f"Tiers: Critical >= {TIER_CRITICAL*100:.0f}% | "
                f"Moderate {TIER_MODERATE*100:.0f}%-{TIER_CRITICAL*100:.0f}% | "
                f"Stable < {TIER_MODERATE*100:.0f}%. "
                f"Latency measured live for this exact request."
            )
        with chart_col:
            render_explanation_panel(shap_labels, shap_vals, shap_colors, shap_raw)
    else:
        with result_col:
            st.info("Set the patient details above, then run the diagnostic evaluation.")
        with chart_col:
            st.caption("A per-patient explanation panel will appear here after you run an evaluation.")

# ---------------------------------------------------------------------------
# TAB 2: Benchmarks
# ---------------------------------------------------------------------------
with tab_bench:
    st.subheader("Clinical Benchmarks & Performance Edge")
    st.caption(
        "Every figure below comes directly from this project's own patient-grouped validation run -- "
        "comparing the production Random Forest ensemble against a Logistic Regression linear baseline "
        "trained and evaluated on the identical split. Nothing here is simulated or estimated."
    )

    auc_lift = RF_METRICS["test_auc"] - LR_METRICS["test_auc"]
    m1, m2, m3 = st.columns(3)
    for col, title, value, sub, accent in [
        (m1, "DISCRIMINATIVE POWER", f"+{auc_lift:.3f} AUC",
         f"Ensemble {RF_METRICS['test_auc']:.3f} vs. Linear baseline {LR_METRICS['test_auc']:.3f}", "#10b981"),
        (m2, "HIGH-RISK RECALL", f"{RF_METRICS['recall_high_risk']*100:.0f}%",
         "Of genuinely high-risk patients correctly flagged", "#06b6d4"),
        (m3, "CV STABILITY", f"{RF_METRICS['cv_auc_mean']:.3f} \u00b1 {RF_METRICS['cv_auc_std']:.3f}",
         "Patient-grouped 5-fold cross-validation AUC", "#f59e0b"),
    ]:
        with col:
            st.markdown(f"""
            <div style="background:#111827;border:1px solid #1f2937;border-radius:12px;padding:20px;">
              <div style="font-size:11px;letter-spacing:1px;color:#6b7280;font-family:'JetBrains Mono',monospace;">{title}</div>
              <div style="font-size:28px;font-weight:800;color:{accent};margin-top:6px;font-family:'JetBrains Mono',monospace;">{value}</div>
              <div style="color:#9ca3af;font-size:13px;margin-top:4px;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("####")
    st.markdown("#### Head-to-head validation metrics")

    rows = [
        ("ROC-AUC (test set)",     f"{RF_METRICS['test_auc']:.3f}",  f"{LR_METRICS['test_auc']:.3f}",
         "Higher AUC means the ensemble ranks at-risk patients above stable ones more reliably across every threshold."),
        ("5-fold CV ROC-AUC",      f"{RF_METRICS['cv_auc_mean']:.3f} (+/- {RF_METRICS['cv_auc_std']:.3f})", "not separately tracked",
         "Confirms the ensemble's score is stable across patient-grouped folds."),
        ("High-risk recall",       f"{RF_METRICS['recall_high_risk']*100:.0f}%", "not separately tracked",
         "Catching high-risk patients is the primary clinical goal -- missing one costs far more than a false alarm."),
        ("High-risk precision",    f"{RF_METRICS['precision_high_risk']*100:.0f}%", "not separately tracked",
         "Deliberate trade-off: the model is tuned toward recall, so most flags are followed up even if not all are truly high-risk."),
        ("High-risk F1",           f"{RF_METRICS['f1_high_risk']:.3f}", "not separately tracked",
         "Balanced precision/recall score for the minority class."),
        ("Stable-class recall",    f"{RF_METRICS['recall_stable']*100:.0f}%", "not separately tracked",
         "Of truly stable patients, this share is correctly identified and spared unnecessary intervention."),
    ]

    table_html = """
    <table style="width:100%;border-collapse:collapse;font-size:13.5px;">
    <thead><tr style="border-bottom:1px solid #1f2937;">
      <th style="text-align:left;padding:10px 8px;color:#9ca3af;font-weight:600;">Metric</th>
      <th style="text-align:left;padding:10px 8px;color:#06b6d4;font-weight:600;">Production RF Ensemble</th>
      <th style="text-align:left;padding:10px 8px;color:#9ca3af;font-weight:600;">LR Linear Baseline</th>
      <th style="text-align:left;padding:10px 8px;color:#9ca3af;font-weight:600;">Clinical / business impact</th>
    </tr></thead><tbody>"""
    for metric, rf_val, lr_val, impact in rows:
        table_html += f"""<tr style="border-bottom:1px solid #1f293780;">
          <td style="padding:10px 8px;color:#e5e7eb;font-family:'JetBrains Mono',monospace;">{metric}</td>
          <td style="padding:10px 8px;color:#10b981;font-weight:700;font-family:'JetBrains Mono',monospace;">{rf_val}</td>
          <td style="padding:10px 8px;color:#6b7280;font-family:'JetBrains Mono',monospace;">{lr_val}</td>
          <td style="padding:10px 8px;color:#9ca3af;">{impact}</td>
        </tr>"""
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.caption(
        "LR entries marked 'not separately tracked' reflect that the validation run measured AUC for both "
        "models but ran the full classification report only on the production model -- shown as-is rather than estimated."
    )

    st.markdown("#### Confusion matrix \u2014 production model (test set, n = {:,})".format(RF_METRICS['n_test']))
    cm_df = pd.DataFrame(
        [[RF_METRICS['tn'], RF_METRICS['fp']], [RF_METRICS['fn'], RF_METRICS['tp']]],
        index=["Actual: stable", "Actual: high risk"],
        columns=["Predicted: stable", "Predicted: high risk"]
    )
    st.table(cm_df)

    st.markdown("#### Global feature importance")
    fig = build_importance_chart(model, schema)
    st.pyplot(fig)

    st.markdown(
        '<div class="disclaimer">This benchmark compares this project\'s own two trained models on an '
        'identical patient-grouped split. It is not a claim of outperforming external published models, '
        'which were not evaluated here. All metrics are directly reproducible from the training notebook.</div>',
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# TAB 3: About
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Problem Statement")
    st.markdown(
        "Hospitals face financial penalties and patient-care consequences when patients are "
        "readmitted within 30 days of discharge. This tool estimates the probability that a "
        "diabetic patient will be readmitted within 30 days, so care teams can prioritize "
        "limited outreach resources toward the patients who need them most."
    )
    st.subheader("Data and Methodology")
    st.markdown(
        "- **Source data:** 101,766 hospital encounters from the Diabetes 130-US Hospitals dataset; "
        "2,423 encounters where the patient expired or entered hospice were removed before modeling.\n"
        "- **Model:** Random Forest classifier (100 trees, max depth 12, class-weight balanced), "
        "compared against a Logistic Regression baseline.\n"
        "- **Validation:** patient-grouped train/test split and patient-grouped 5-fold cross-validation, "
        "so a patient's multiple hospital stays never appear on both sides of a split.\n"
        "- **Explainability:** global feature importance plus per-patient SHAP values with a dual-panel "
        "visualizer (SHAP attribution + input range chart), computed live for every evaluation.\n"
        "- **Risk tiers:** derived from the validation set's actual score distribution (percentile-based), "
        "not arbitrary fixed cutoffs.\n"
        "- **Latency:** inference and explanation timings shown after each evaluation are measured live "
        "on this server for that exact request."
    )
    st.subheader("Known Simplifications")
    st.markdown(
        "- Secondary and tertiary diagnosis codes are held at baseline to keep the form usable.\n"
        "- The benchmark tab compares this project's two own trained models only.\n"
        "- Hospital-specific cost figures are not part of this tool."
    )
    st.markdown(
        '<div class="disclaimer">This is a demonstration project built on a public research dataset for '
        'portfolio purposes. It is not a certified clinical decision-support system and must not be used '
        'for real patient care decisions.</div>',
        unsafe_allow_html=True
    )
