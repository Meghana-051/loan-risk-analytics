import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Loan Risk Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv("data/loan_data.csv")

# =========================================================
# CLEAN DATA
# =========================================================
cleaned_df = df.copy()

fill_mode_cols = [
    "Gender", "Married", "Dependents", "Self_Employed",
    "Credit_History", "Loan_Amount_Term"
]

for col in fill_mode_cols:
    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])

cleaned_df["LoanAmount"] = cleaned_df["LoanAmount"].fillna(cleaned_df["LoanAmount"].median())

text_cols = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area", "Loan_Status"
]

for col in text_cols:
    cleaned_df[col] = cleaned_df[col].astype(str).str.strip()

cleaned_df["TotalIncome"] = cleaned_df["ApplicantIncome"] + cleaned_df["CoapplicantIncome"]

cleaned_df["IncomeCategory"] = pd.cut(
    cleaned_df["ApplicantIncome"],
    bins=[0, 3000, 6000, 10000, 1000000],
    labels=["Low", "Medium", "High", "Very High"],
    include_lowest=True
)

cleaned_df["LoanAmountCategory"] = pd.cut(
    cleaned_df["LoanAmount"],
    bins=[0, 100, 200, 300, 10000],
    labels=["Small", "Moderate", "Large", "Very Large"],
    include_lowest=True
)

def assign_risk(row):
    score = 0
    if row["Credit_History"] == 0:
        score += 3
    if row["LoanAmount"] > cleaned_df["LoanAmount"].median():
        score += 1
    if row["ApplicantIncome"] < cleaned_df["ApplicantIncome"].median():
        score += 1
    if row["Loan_Status"] == "N":
        score += 2

    if score >= 4:
        return "High Risk"
    elif score >= 2:
        return "Medium Risk"
    return "Low Risk"

cleaned_df["RiskSegment"] = cleaned_df.apply(assign_risk, axis=1)

# =========================================================
# LOAD MODEL
# =========================================================
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Dashboard Controls")

theme_mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"], index=0)
chart_theme = "plotly_dark" if theme_mode == "Dark" else "plotly_white"

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard Overview", "Advanced Analytics", "Model Insights", "Prediction Studio"]
)

gender_filter = st.sidebar.multiselect(
    "Gender",
    options=sorted(cleaned_df["Gender"].unique()),
    default=sorted(cleaned_df["Gender"].unique())
)

education_filter = st.sidebar.multiselect(
    "Education",
    options=sorted(cleaned_df["Education"].unique()),
    default=sorted(cleaned_df["Education"].unique())
)

property_filter = st.sidebar.multiselect(
    "Property Area",
    options=sorted(cleaned_df["Property_Area"].unique()),
    default=sorted(cleaned_df["Property_Area"].unique())
)

married_filter = st.sidebar.multiselect(
    "Marital Status",
    options=sorted(cleaned_df["Married"].unique()),
    default=sorted(cleaned_df["Married"].unique())
)

self_emp_filter = st.sidebar.multiselect(
    "Self Employed",
    options=sorted(cleaned_df["Self_Employed"].unique()),
    default=sorted(cleaned_df["Self_Employed"].unique())
)

dependents_filter = st.sidebar.multiselect(
    "Dependents",
    options=sorted(cleaned_df["Dependents"].unique()),
    default=sorted(cleaned_df["Dependents"].unique())
)

credit_filter = st.sidebar.multiselect(
    "Credit History",
    options=sorted(cleaned_df["Credit_History"].unique()),
    default=sorted(cleaned_df["Credit_History"].unique())
)

loan_status_filter = st.sidebar.multiselect(
    "Loan Status",
    options=sorted(cleaned_df["Loan_Status"].unique()),
    default=sorted(cleaned_df["Loan_Status"].unique())
)

risk_filter = st.sidebar.multiselect(
    "Risk Segment",
    options=sorted(cleaned_df["RiskSegment"].unique()),
    default=sorted(cleaned_df["RiskSegment"].unique())
)

income_min = int(cleaned_df["ApplicantIncome"].min())
income_max = int(cleaned_df["ApplicantIncome"].max())
income_range = st.sidebar.slider(
    "Applicant Income Range",
    min_value=income_min,
    max_value=income_max,
    value=(income_min, income_max)
)

loan_min = int(cleaned_df["LoanAmount"].min())
loan_max = int(cleaned_df["LoanAmount"].max())
loan_range = st.sidebar.slider(
    "Loan Amount Range",
    min_value=loan_min,
    max_value=loan_max,
    value=(loan_min, loan_max)
)

term_filter = st.sidebar.multiselect(
    "Loan Term",
    options=sorted(cleaned_df["Loan_Amount_Term"].unique()),
    default=sorted(cleaned_df["Loan_Amount_Term"].unique())
)

status_chart_type = st.sidebar.selectbox(
    "Loan Status Chart Type",
    ["Bar", "Pie", "Donut"],
    index=0
)

show_data_labels = st.sidebar.checkbox("Show Data Labels", value=True)

preset = st.sidebar.selectbox(
    "Quick Preset",
    ["All Applicants", "High Risk Applicants", "Good Credit Applicants", "Urban Applicants", "High Income Applicants"]
)

# =========================================================
# APPLY PRESET
# =========================================================
preset_df = cleaned_df.copy()

if preset == "High Risk Applicants":
    preset_df = preset_df[preset_df["RiskSegment"] == "High Risk"]
elif preset == "Good Credit Applicants":
    preset_df = preset_df[preset_df["Credit_History"] == 1]
elif preset == "Urban Applicants":
    preset_df = preset_df[preset_df["Property_Area"] == "Urban"]
elif preset == "High Income Applicants":
    preset_df = preset_df[preset_df["ApplicantIncome"] >= preset_df["ApplicantIncome"].median()]

# =========================================================
# APPLY FILTERS
# =========================================================
filtered_df = preset_df[
    (preset_df["Gender"].isin(gender_filter)) &
    (preset_df["Education"].isin(education_filter)) &
    (preset_df["Property_Area"].isin(property_filter)) &
    (preset_df["Married"].isin(married_filter)) &
    (preset_df["Self_Employed"].isin(self_emp_filter)) &
    (preset_df["Dependents"].isin(dependents_filter)) &
    (preset_df["Credit_History"].isin(credit_filter)) &
    (preset_df["Loan_Status"].isin(loan_status_filter)) &
    (preset_df["RiskSegment"].isin(risk_filter)) &
    (preset_df["ApplicantIncome"].between(income_range[0], income_range[1])) &
    (preset_df["LoanAmount"].between(loan_range[0], loan_range[1])) &
    (preset_df["Loan_Amount_Term"].isin(term_filter))
]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

# =========================================================
# THEME COLORS
# =========================================================
if theme_mode == "Dark":
    page_bg = "#0f172a"
    text_color = "#f8fafc"
    subtext_color = "#cbd5e1"
    card_bg = "#111827"
    border_color = "#334155"
    info_bg = "#172554"
    info_border = "#3b82f6"
else:
    page_bg = "#f4f7fb"
    text_color = "#0f172a"
    subtext_color = "#64748b"
    card_bg = "#ffffff"
    border_color = "#e2e8f0"
    info_bg = "#eff6ff"
    info_border = "#2563eb"

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(f"""
<style>
.stApp {{
    background: {page_bg};
}}

.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 95rem;
}}

.hero {{
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 50%, #2563eb 100%);
    padding: 34px 36px;
    border-radius: 24px;
    color: white;
    box-shadow: 0 14px 38px rgba(37, 99, 235, 0.20);
    margin-bottom: 20px;
}}

.hero-badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.18);
    font-size: 13px;
    margin-bottom: 14px;
}}

.hero-title {{
    font-size: 38px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 8px;
}}

.hero-subtitle {{
    font-size: 16px;
    color: #dbeafe;
    max-width: 900px;
}}

.metric-card {{
    background: {card_bg};
    border-radius: 20px;
    padding: 18px 20px;
    border: 1px solid {border_color};
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    margin-bottom: 10px;
}}

.metric-label {{
    font-size: 13px;
    color: {subtext_color};
    margin-bottom: 8px;
}}

.metric-value {{
    font-size: 28px;
    font-weight: 800;
    color: {text_color};
}}

.section-title {{
    font-size: 22px;
    font-weight: 750;
    color: {text_color};
    margin-bottom: 4px;
}}

.section-subtitle {{
    color: {subtext_color};
    margin-bottom: 14px;
    font-size: 14px;
}}

.insight-box {{
    background: {info_bg};
    border: 1px solid {info_border};
    border-left: 5px solid {info_border};
    border-radius: 16px;
    padding: 16px;
    color: {text_color};
}}

.result-box {{
    padding: 18px;
    border-radius: 18px;
    font-size: 18px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 10px;
}}

.approve-box {{
    background: #ecfdf5;
    color: #065f46;
    border: 1px solid #a7f3d0;
}}

.risk-box {{
    background: #fef2f2;
    color: #991b1b;
    border: 1px solid #fecaca;
}}

.small-chip {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: #eef2ff;
    color: #3730a3;
    border: 1px solid #c7d2fe;
    font-size: 12px;
    margin-right: 8px;
    margin-top: 8px;
}}

.summary-strip {{
    background: {card_bg};
    border: 1px solid {border_color};
    border-radius: 16px;
    padding: 14px 18px;
    margin: 14px 0 18px 0;
    color: {text_color};
}}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="hero-badge">Banking Analytics • Machine Learning • Risk Intelligence</div>
    <div class="hero-title">Loan Risk Analytics and Default Prediction System</div>
    <div class="hero-subtitle">
        Professional decision-support dashboard for analyzing applicant trends, lending risk,
        financial behavior, model drivers, and approval probability.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<span class="small-chip">Interactive Filters</span>
<span class="small-chip">Executive KPIs</span>
<span class="small-chip">Risk Segmentation</span>
<span class="small-chip">Prediction Studio</span>
<span class="small-chip">Feature Importance</span>
<span class="small-chip">Download Data</span>
""", unsafe_allow_html=True)

# =========================================================
# KPI VALUES
# =========================================================
total_applicants = filtered_df.shape[0]
avg_income = filtered_df["ApplicantIncome"].mean()
avg_loan = filtered_df["LoanAmount"].mean()
approval_rate = (filtered_df["Loan_Status"] == "Y").mean() * 100
approved_count = (filtered_df["Loan_Status"] == "Y").sum()
risky_count = (filtered_df["Loan_Status"] == "N").sum()
good_credit_count = (filtered_df["Credit_History"] == 1).sum()

risk_counts = filtered_df["RiskSegment"].value_counts()
low_risk_count = int(risk_counts.get("Low Risk", 0))
medium_risk_count = int(risk_counts.get("Medium Risk", 0))
high_risk_count = int(risk_counts.get("High Risk", 0))

# =========================================================
# SUMMARY STRIP
# =========================================================
active_filters = []
if len(gender_filter) != len(cleaned_df["Gender"].unique()):
    active_filters.append("Gender")
if len(education_filter) != len(cleaned_df["Education"].unique()):
    active_filters.append("Education")
if len(property_filter) != len(cleaned_df["Property_Area"].unique()):
    active_filters.append("Property Area")
if len(married_filter) != len(cleaned_df["Married"].unique()):
    active_filters.append("Marital Status")
if len(self_emp_filter) != len(cleaned_df["Self_Employed"].unique()):
    active_filters.append("Self Employed")
if len(credit_filter) != len(cleaned_df["Credit_History"].unique()):
    active_filters.append("Credit History")
if len(risk_filter) != len(cleaned_df["RiskSegment"].unique()):
    active_filters.append("Risk Segment")

filter_text = ", ".join(active_filters) if active_filters else "Default view"

st.markdown(f"""
<div class="summary-strip">
    <b>Current Page:</b> {page} &nbsp;&nbsp; | &nbsp;&nbsp;
    <b>Preset:</b> {preset} &nbsp;&nbsp; | &nbsp;&nbsp;
    <b>Records Shown:</b> {total_applicants} &nbsp;&nbsp; | &nbsp;&nbsp;
    <b>Active Filters:</b> {filter_text}
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPI CARDS
# =========================================================
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Applicants</div>
        <div class="metric-value">{total_applicants}</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Income</div>
        <div class="metric-value">{avg_income:.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Loan Amount</div>
        <div class="metric-value">{avg_loan:.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Approval Rate</div>
        <div class="metric-value">{approval_rate:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Good Credit Profiles</div>
        <div class="metric-value">{good_credit_count}</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# DOWNLOAD
# =========================================================
st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_df.to_csv(index=False).encode("utf-8"),
    file_name="filtered_loan_data.csv",
    mime="text/csv"
)

# =========================================================
# PAGE: DASHBOARD OVERVIEW
# =========================================================
if page == "Dashboard Overview":
    st.markdown("## Executive Overview")
    st.caption("High-level portfolio view, approval trends, and risk segmentation.")

    c1, c2 = st.columns(2)

    with c1:
        loan_status_counts = filtered_df["Loan_Status"].value_counts().reset_index()
        loan_status_counts.columns = ["Loan_Status", "Count"]
        text_arg = "Count" if show_data_labels and status_chart_type == "Bar" else None

        if status_chart_type == "Bar":
            fig1 = px.bar(
                loan_status_counts,
                x="Loan_Status",
                y="Count",
                color="Loan_Status",
                text=text_arg,
                title="Loan Status Distribution",
                template=chart_theme
            )
        elif status_chart_type == "Pie":
            fig1 = px.pie(
                loan_status_counts,
                names="Loan_Status",
                values="Count",
                title="Loan Status Distribution",
                template=chart_theme
            )
        else:
            fig1 = px.pie(
                loan_status_counts,
                names="Loan_Status",
                values="Count",
                hole=0.5,
                title="Loan Status Distribution",
                template=chart_theme
            )
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        credit_status = filtered_df.groupby(["Credit_History", "Loan_Status"]).size().reset_index(name="Count")
        fig2 = px.bar(
            credit_status,
            x="Credit_History",
            y="Count",
            color="Loan_Status",
            barmode="group",
            text="Count" if show_data_labels else None,
            title="Credit History vs Loan Status",
            template=chart_theme
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        property_status = filtered_df.groupby(["Property_Area", "Loan_Status"]).size().reset_index(name="Count")
        fig3 = px.bar(
            property_status,
            x="Property_Area",
            y="Count",
            color="Loan_Status",
            barmode="group",
            text="Count" if show_data_labels else None,
            title="Property Area vs Loan Status",
            template=chart_theme
        )
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        risk_seg = filtered_df["RiskSegment"].value_counts().reset_index()
        risk_seg.columns = ["RiskSegment", "Count"]
        fig4 = px.pie(
            risk_seg,
            names="RiskSegment",
            values="Count",
            hole=0.45,
            title="Risk Segmentation",
            template=chart_theme
        )
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("Low Risk Applicants", low_risk_count)
    with r2:
        st.metric("Medium Risk Applicants", medium_risk_count)
    with r3:
        st.metric("High Risk Applicants", high_risk_count)

    top_property = filtered_df["Property_Area"].mode()[0]
    top_education = filtered_df["Education"].mode()[0]
    top_income_segment = filtered_df["IncomeCategory"].mode()[0]

    st.markdown(f"""
    <div class="insight-box">
        <b>Portfolio Snapshot</b><br><br>
        • Approved applications: <b>{approved_count}</b><br>
        • Risky / rejected applications: <b>{risky_count}</b><br>
        • Most common property area: <b>{top_property}</b><br>
        • Most common education category: <b>{top_education}</b><br>
        • Most common income segment: <b>{top_income_segment}</b>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# PAGE: ADVANCED ANALYTICS
# =========================================================
elif page == "Advanced Analytics":
    st.markdown("## Advanced Analytics")
    st.caption("Deep-dive views across financial, demographic, and outlier patterns.")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig5 = px.histogram(
            filtered_df,
            x="ApplicantIncome",
            nbins=20,
            title="Applicant Income Distribution",
            template=chart_theme
        )
        fig5.update_layout(height=420)
        st.plotly_chart(fig5, use_container_width=True)

    with r1c2:
        fig6 = px.histogram(
            filtered_df,
            x="LoanAmount",
            nbins=20,
            title="Loan Amount Distribution",
            template=chart_theme
        )
        fig6.update_layout(height=420)
        st.plotly_chart(fig6, use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        fig7 = px.box(
            filtered_df,
            x="Loan_Status",
            y="ApplicantIncome",
            color="Loan_Status",
            title="Income by Loan Status",
            template=chart_theme
        )
        fig7.update_layout(height=420)
        st.plotly_chart(fig7, use_container_width=True)

    with r2c2:
        fig8 = px.box(
            filtered_df,
            x="Property_Area",
            y="LoanAmount",
            color="Property_Area",
            title="Loan Amount by Property Area",
            template=chart_theme
        )
        fig8.update_layout(height=420)
        st.plotly_chart(fig8, use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        gender_status = filtered_df.groupby(["Gender", "Loan_Status"]).size().reset_index(name="Count")
        fig9 = px.bar(
            gender_status,
            x="Gender",
            y="Count",
            color="Loan_Status",
            barmode="group",
            text="Count" if show_data_labels else None,
            title="Gender vs Loan Status",
            template=chart_theme
        )
        fig9.update_layout(height=420)
        st.plotly_chart(fig9, use_container_width=True)

    with r3c2:
        education_status = filtered_df.groupby(["Education", "Loan_Status"]).size().reset_index(name="Count")
        fig10 = px.bar(
            education_status,
            x="Education",
            y="Count",
            color="Loan_Status",
            barmode="group",
            text="Count" if show_data_labels else None,
            title="Education vs Loan Status",
            template=chart_theme
        )
        fig10.update_layout(height=420)
        st.plotly_chart(fig10, use_container_width=True)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        married_status = filtered_df.groupby(["Married", "Loan_Status"]).size().reset_index(name="Count")
        fig11 = px.bar(
            married_status,
            x="Married",
            y="Count",
            color="Loan_Status",
            barmode="group",
            text="Count" if show_data_labels else None,
            title="Marital Status vs Loan Status",
            template=chart_theme
        )
        fig11.update_layout(height=420)
        st.plotly_chart(fig11, use_container_width=True)

    with r4c2:
        self_emp_status = filtered_df.groupby(["Self_Employed", "Loan_Status"]).size().reset_index(name="Count")
        fig12 = px.bar(
            self_emp_status,
            x="Self_Employed",
            y="Count",
            color="Loan_Status",
            barmode="group",
            text="Count" if show_data_labels else None,
            title="Self Employed vs Loan Status",
            template=chart_theme
        )
        fig12.update_layout(height=420)
        st.plotly_chart(fig12, use_container_width=True)

    r5c1, r5c2 = st.columns(2)
    with r5c1:
        dep_status = filtered_df.groupby(["Dependents", "Loan_Status"]).size().reset_index(name="Count")
        fig13 = px.bar(
            dep_status,
            x="Dependents",
            y="Count",
            color="Loan_Status",
            barmode="group",
            text="Count" if show_data_labels else None,
            title="Dependents vs Loan Status",
            template=chart_theme
        )
        fig13.update_layout(height=420)
        st.plotly_chart(fig13, use_container_width=True)

    with r5c2:
        fig14 = px.scatter(
            filtered_df,
            x="TotalIncome",
            y="LoanAmount",
            color="Loan_Status",
            size="ApplicantIncome",
            hover_data=["Gender", "Education", "Property_Area"],
            title="Total Income vs Loan Amount",
            template=chart_theme
        )
        fig14.update_layout(height=420)
        st.plotly_chart(fig14, use_container_width=True)

# =========================================================
# PAGE: MODEL INSIGHTS
# =========================================================
elif page == "Model Insights":
    st.markdown("## Model Insights")
    st.caption("Understand feature importance and variable relationships.")

    c1, c2 = st.columns(2)

    with c1:
        feature_names = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area"
        ]

        if hasattr(model, "feature_importances_"):
            feature_importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            fig15 = px.bar(
                feature_importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance",
                template=chart_theme
            )
            fig15.update_layout(height=420)
            st.plotly_chart(fig15, use_container_width=True)
        else:
            st.info("Feature importance is not available for this model.")

    with c2:
        corr_df = filtered_df[[
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "TotalIncome"
        ]].copy()

        corr = corr_df.corr(numeric_only=True)

        heatmap = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                text=corr.round(2).values,
                texttemplate="%{text}",
                colorscale="Blues"
            )
        )
        heatmap.update_layout(title="Correlation Heatmap", template=chart_theme, height=420)
        st.plotly_chart(heatmap, use_container_width=True)

# =========================================================
# PAGE: PREDICTION STUDIO
# =========================================================
elif page == "Prediction Studio":
    st.markdown("## Prediction Studio")
    st.caption("Enter applicant information to estimate approval probability and lending risk.")

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)

        with c2:
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
            loan_amount = st.number_input("Loan Amount", min_value=0, value=120)
            loan_amount_term = st.selectbox("Loan Amount Term", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480])
            credit_history = st.selectbox("Credit History", [1.0, 0.0])
            property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])

        submit_prediction = st.form_submit_button("Run Risk Prediction")

    if submit_prediction:
        input_data = pd.DataFrame({
            "Gender": [gender],
            "Married": [married],
            "Dependents": [dependents],
            "Education": [education],
            "Self_Employed": [self_employed],
            "ApplicantIncome": [applicant_income],
            "CoapplicantIncome": [coapplicant_income],
            "LoanAmount": [loan_amount],
            "Loan_Amount_Term": [loan_amount_term],
            "Credit_History": [credit_history],
            "Property_Area": [property_area]
        })

        for col in ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]:
            input_data[col] = label_encoders[col].transform(input_data[col])

        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        loan_status_label = label_encoders["Loan_Status"].inverse_transform([prediction])[0]
        class_labels = label_encoders["Loan_Status"].inverse_transform(model.classes_)
        prob_map = dict(zip(class_labels, prediction_proba))

        approval_prob = prob_map.get("Y", 0) * 100
        risk_prob = prob_map.get("N", 0) * 100

        st.markdown("### Prediction Result")

        if loan_status_label == "Y":
            st.markdown(
                '<div class="result-box approve-box">Loan likely to be Approved</div>',
                unsafe_allow_html=True
            )
            recommendation = "Recommended for approval based on the current applicant profile."
        else:
            st.markdown(
                '<div class="result-box risk-box">Loan is Risky / Likely Rejected</div>',
                unsafe_allow_html=True
            )
            recommendation = "Requires manual review due to elevated lending risk."

        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Approval Probability", f"{approval_prob:.2f}%")
        with p2:
            st.metric("Risk Probability", f"{risk_prob:.2f}%")
        with p3:
            st.metric("Total Income", f"{applicant_income + coapplicant_income}")

        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=approval_prob,
            title={"text": "Approval Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fecaca"},
                    {"range": [40, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#bbf7d0"}
                ]
            }
        ))
        gauge_fig.update_layout(template=chart_theme, height=350)
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
            <b>Business Recommendation</b><br><br>
            {recommendation}
        </div>
        """, unsafe_allow_html=True)