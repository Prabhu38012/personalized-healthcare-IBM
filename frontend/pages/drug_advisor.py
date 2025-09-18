import streamlit as st
import os
import pandas as pd
from typing import List, Dict, Any
import io
import matplotlib.pyplot as plt
import networkx as nx

from utils.api_client import HealthcareAPI
from components.auth import (
    init_session_state, check_session_validity, render_login_page,
    render_user_info, require_auth, get_auth_headers
)
from utils.report_export import build_drug_advisor_pdf


# Optional: disable this page via environment flag
if os.getenv("HIDE_DRUG_ADVISOR", "0") in ("1", "true", "True"):
    st.info("This page is disabled because all features are integrated into Medical Report Analysis.")
    st.stop()

# Initialize auth and API client
init_session_state()
api = st.session_state.get('api_client') or HealthcareAPI()
api.set_auth_headers(get_auth_headers())

st.title("💊 AI Medical Prescription Verification")
st.caption("IBM Watson enriched + Hugging Face NER • Interactions • Age-specific dosage • Alternatives")

# Light-weight CSS for polished visuals
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(90deg, #f0fff4 0%, #ecfeff 100%);
        border: 1px solid #e6f6ef;
        padding: 16px 20px; border-radius: 12px; margin-bottom: 14px;
    }
    .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; }
    .ok { background:#dcfce7; color:#166534; border:1px solid #bbf7d0; }
    .warn { background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; }
    .card { border: 1px solid #edf2f7; border-radius: 12px; padding: 16px; background: #fff; }
    .tags span { display:inline-block; margin: 2px 6px 0 0; padding: 6px 10px; border-radius: 999px; background:#f1f5f9; font-size: 12px; }
    .section-title { font-size: 18px; font-weight: 600; margin-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero header
with st.container():
    st.markdown("""
    <div class="hero">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:10px;">
            <div>
                <div style="font-size:20px; font-weight:600; color:#065f46;">Drug Advisor</div>
                <div style="font-size:13px; color:#0f766e;">Extract • Interactions • Dosage • Alternatives</div>
            </div>
            <div id="health-badge"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar: user info and health
with st.sidebar:
    render_user_info()
    health = api.check_backend_health()
    if health and isinstance(health, dict):
        st.markdown('<span class="badge ok">Backend connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge warn">Backend offline (check port 8002)</span>', unsafe_allow_html=True)

# Tabs for features
extract_tab, interact_tab, dosage_tab, alt_tab = st.tabs([
    "Extract Drug Info (NLP)",
    "Interaction Checker",
    "Dosage Recommendation",
    "Alternative Suggestions",
])

with extract_tab:
    st.markdown('<div class="section-title">Extract structured drug details from text</div>', unsafe_allow_html=True)
    sample = "Patient prescribed ibuprofen 200mg TID and amoxicillin 500mg BID for 7 days."
    # Optional file upload to OCR a prescription and prefill text
    up1, up2 = st.columns([1.2, 1])
    with up1:
        uploaded = st.file_uploader("Choose medical text source (PDF or image)", type=["pdf","png","jpg","jpeg","tiff","bmp","webp"], accept_multiple_files=False)
        pname = st.text_input("Patient Name (optional)", value=st.session_state.get('user_name',''))
        if st.button("📄 Analyze Report & Prefill Text", use_container_width=True, key="analyze_file_btn"):
            if not uploaded:
                st.warning("Please choose a PDF or image file.")
            else:
                with st.spinner("Uploading and extracting text from file…"):
                    file_tuple = (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream")
                    pr = api.upload_prescription(file_tuple, pname or "Patient")
                if pr and pr.get('extracted_text'):
                    combined = "\n".join(pr.get('extracted_text', []))[:8000]
                    st.session_state['drug_text'] = combined
                    st.session_state['patient_name'] = pname or "Patient"
                    st.success("Text extracted. Prefilled below.")
                    st.rerun()
    with up2:
        st.markdown("<div class='card' style='min-height:180px'><b>Tips</b><br>• Include drug names, strengths, and frequencies (e.g., 500mg BID).<br>• Use clear typed text for best results.<br>• Watson enrichment requires credentials.</div>", unsafe_allow_html=True)
    # Maintain a persistent textarea value
    if 'drug_text' not in st.session_state:
        st.session_state['drug_text'] = sample
    # Patient chip and preview
    pn = st.session_state.get('patient_name')
    if pn:
        st.markdown(f"<div class='badge ok'>Patient: {pn}</div>", unsafe_allow_html=True)
    with st.expander("Preview extracted text", expanded=False):
        st.code((st.session_state.get('drug_text') or '')[:600] + ('...' if len(st.session_state.get('drug_text',''))>600 else ''), language='text')

    text = st.text_area("Paste prescription/doctor note text", value=st.session_state['drug_text'], height=180, placeholder="Paste prescription text here…")
    # Age/Weight/Condition input for downstream dosage guidance
    col_age, col_wt, col_cond, col_btn = st.columns([0.8,0.8,1.2,1])
    with col_age:
        age_for_dosage = st.number_input("Age (years)", min_value=0, max_value=120, value=int(st.session_state.get('user_age') or 30), key="extract_age_years")
    with col_wt:
        weight_for_dosage = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=0.0, step=0.1, help="Optional. Used for pediatric mg/kg dosing.", key="extract_weight_kg")
    with col_cond:
        condition_context = st.text_input("Condition (optional)", value="", placeholder="e.g., diabetes, renal impairment", key="extract_condition")
    with col_btn:
        go = st.button("🔎 Extract", type="primary", use_container_width=True, key="extract_btn")
    if go:
        with st.spinner("Extracting entities with Hugging Face (fallback: regex) and optionally enriching with IBM Watson…"):
            res = api.extract_drug_info(text)
        if res:
            extracted = res.get('extracted', [])
            drugs_df = pd.DataFrame(extracted)

            # Fetch interactions and alternatives for dashboard stats
            drug_list = [d.get('name') for d in extracted if d.get('name')]
            inter_res = api.check_interactions(drug_list) if len(drug_list) >= 2 else {"interactions": []}
            interactions = inter_res.get('interactions', []) if inter_res else []

            # Age-specific dosage recommendations (batch over extracted drugs)
            dosage_recs: List[Dict[str, Any]] = []
            for dname in drug_list[:8]:  # limit calls for performance
                try:
                    # Pass optional weight/condition if provided
                    w = float(weight_for_dosage) if weight_for_dosage and weight_for_dosage > 0 else None
                    cond = condition_context.strip() or None
                    dres = api.dosage_recommendation(dname, int(age_for_dosage), w, cond)
                    if dres:
                        dosage_recs.append(dres)
                except Exception:
                    pass

            # Simple alternatives by each drug (category based)
            alternatives: List[Dict[str, Any]] = []
            for d in drug_list[:6]:  # limit calls
                alt = api.alternative_suggestions(d)
                if alt and alt.get('alternatives'):
                    alternatives.append(alt)

            # Dosage warnings: if normalization present and frequency_per_day is too high for common drugs (heuristic)
            warnings_count = 0
            for d in extracted:
                per_day = d.get('frequency_per_day')
                if isinstance(per_day, (int, float)) and per_day > 4:
                    warnings_count += 1

            # Summary dashboard
            st.markdown("#### Summary Dashboard")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Drugs", len(drug_list))
            col2.metric("Interactions Flagged", len(interactions))
            col3.metric("Safer Alternatives", sum(len(a.get('alternatives', [])) for a in alternatives))
            col4.metric("Dosage Warnings", warnings_count)

            st.divider()

            # Tabs for detailed views
            t1, t2, t3, t4 = st.tabs(["Extracted Drug Info", "Interactions", "Dosage Guidance", "Alternatives"])

            with t1:
                st.markdown("##### Extracted Drug Info")
                if not drugs_df.empty:
                    view_df = drugs_df.copy()
                    # Add confidence % and source label
                    if 'score' in view_df.columns:
                        view_df['confidence_%'] = (view_df['score'].astype(float) * 100).round(0)
                    if 'source' not in view_df.columns:
                        view_df['source'] = view_df.apply(lambda r: 'HuggingFace NER' if pd.notnull(r.get('score')) else 'regex', axis=1)
                    cols = [c for c in [
                        'name','strength','frequency','duration','confidence_%','source',
                        'strength_mg','frequency_per_day','duration_days'
                    ] if c in view_df.columns]
                    st.dataframe(view_df[cols], use_container_width=True)
                    # Wordcloud visualization of detected drugs (if library available)
                    try:
                        from wordcloud import WordCloud
                        freq = {}
                        for n in view_df['name'].dropna().tolist():
                            freq[n] = freq.get(n, 0) + 1
                        if freq:
                            wc = WordCloud(width=600, height=250, background_color='white')
                            wc_img = wc.generate_from_frequencies(freq)
                            fig_wc, ax_wc = plt.subplots(figsize=(6,2.8))
                            ax_wc.imshow(wc_img, interpolation='bilinear')
                            ax_wc.axis('off')
                            st.pyplot(fig_wc, use_container_width=True)
                    except Exception:
                        pass
                else:
                    st.info("No drug entities detected.")

            with t2:
                st.markdown("##### Interactions")
                if interactions:
                    idf = pd.DataFrame(interactions)
                    st.dataframe(idf, use_container_width=True)

                    # Network graph
                    st.markdown("###### Interaction Network")
                    G = nx.Graph()
                    for it in interactions:
                        pair = it.get('pair', [])
                        if len(pair) == 2:
                            sev = (it.get('severity') or 'unknown').lower()
                            color = {'mild':'#22c55e','moderate':'#f59e0b','severe':'#ef4444'}.get(sev, '#94a3b8')
                            G.add_edge(pair[0], pair[1], color=color)
                    pos = nx.spring_layout(G, seed=42)
                    edge_colors = [G[u][v]['color'] for u,v in G.edges()]
                    fig, ax = plt.subplots(figsize=(6,4))
                    nx.draw_networkx(G, pos, with_labels=True, node_color='#bfdbfe', edge_color=edge_colors, ax=ax)
                    ax.axis('off')
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("No interactions flagged.")

            with t3:
                st.markdown("##### Dosage Guidance & Compliance")
                # Display age-specific dosage recommendations for extracted drugs
                if dosage_recs:
                    drows = []
                    for item in dosage_recs:
                        drows.append({
                            'drug': item.get('drug'),
                            'age': item.get('age'),
                            'age_band': item.get('age_band'),
                            'recommendation': item.get('recommendation'),
                            'source': item.get('source'),
                            'calculated_dose_mg': item.get('calculated_dose_mg'),
                            'rounded_dose_mg': item.get('rounded_dose_mg')
                        })
                    st.dataframe(pd.DataFrame(drows), use_container_width=True)
                    st.caption("Age-specific recommendations are heuristic or label-based. Verify clinically.")
                # Simple chart: strength_mg per drug
                if drugs_df.empty or 'strength_mg' not in drugs_df.columns:
                    st.info("No normalized strength available to chart.")
                else:
                    plot_df = drugs_df[['name','strength_mg']].dropna()
                    if plot_df.empty:
                        st.info("No normalized strength available to chart.")
                    else:
                        fig, ax = plt.subplots(figsize=(6,3))
                        ax.bar(plot_df['name'], plot_df['strength_mg'], color='#60a5fa')
                        ax.set_ylabel('Dose (mg)')
                        ax.set_xlabel('Drug')
                        ax.set_title('Prescribed Dosage (mg)')
                        plt.xticks(rotation=20)
                        st.pyplot(fig, use_container_width=True)
                st.caption("Note: Safe ranges vary by patient and indication; consult validated references.")

            with t4:
                st.markdown("##### Safer Alternatives")
                if alternatives:
                    rows = []
                    for alt in alternatives:
                        rows.append({
                            'drug': alt.get('drug'),
                            'category': alt.get('category'),
                            'alternatives': ', '.join(alt.get('alternatives', []))
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)
                else:
                    st.info("No alternatives available.")

            # Export Button
            st.markdown("#### Export Report")
            patient = {'name': st.session_state.get('user_name','-'), 'age': st.session_state.get('user_age','-')}
            dosage_notes: List[str] = []
            pdf_bytes = build_drug_advisor_pdf(patient, extracted, interactions, dosage_notes, alternatives)
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="drug_advisor_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            # Watson info
            wat = res.get('watson', {})
            with st.expander("IBM Watson Context", expanded=False):
                if wat and wat.get('used'):
                    st.json(wat.get('data', {}))
                else:
                    st.caption(f"Not used: {wat.get('reason', 'credentials missing')}")

with interact_tab:
    st.markdown('<div class="section-title">Check for drug-drug interactions</div>', unsafe_allow_html=True)
    st.caption("Curated rules → openFDA label scan")
    drugs = st.text_input("Enter drugs (comma-separated)", value="ibuprofen, aspirin, amoxicillin", placeholder="e.g., metformin, lisinopril, spironolactone", key="interact_drug_list")
    parsed: List[str] = [d.strip() for d in drugs.split(',') if d.strip()]
    btn = st.button("🧪 Check Interactions", use_container_width=True, key="chk_interact")
    if btn:
        if len(parsed) < 2:
            st.warning("Enter at least two drugs")
        else:
            with st.spinner("Checking interactions…"):
                res = api.check_interactions(parsed)
            if res:
                st.info(res.get('summary'))
                inter = res.get('interactions', [])
                if inter:
                    df = pd.DataFrame(inter)
                    st.markdown("#### Potential Interactions")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("No interactions found in rules or label scan. This is not a guarantee of safety.")

with dosage_tab:
    st.markdown('<div class="section-title">Age-specific dosage guidance</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1.5,0.8,1.2])
    with col1:
        drug = st.text_input("Drug name", value="ibuprofen", placeholder="e.g., amoxicillin", key="dosage_drug_name")
    with col2:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, key="dosage_age_years")
    with col3:
        weight_single = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=0.0, step=0.1, key="dosage_weight_kg")
    condition_single = st.text_input("Condition (optional)", value="", placeholder="e.g., hypertension, pregnancy", key="dosage_condition")
    if st.button("📋 Get Dosage", use_container_width=True, key="dosage_btn"):
        with st.spinner("Fetching dosage guidance (heuristics → label fallback)…"):
            w = float(weight_single) if weight_single and weight_single > 0 else None
            cond = condition_single.strip() or None
            res = api.dosage_recommendation(drug, int(age), w, cond)
        if res:
            st.markdown("#### Recommendation")
            st.markdown(f"<div class='card'>{res.get('recommendation', 'No guidance')}</div>", unsafe_allow_html=True)
            st.caption(f"Source: {res.get('source', 'unknown')} • Age Band: {res.get('age_band', '-')}")
            # Show computed pediatric dose and warnings if available
            cd = res.get('calculated_dose_mg')
            rd = res.get('rounded_dose_mg')
            warns = res.get('warnings', [])
            if cd or rd or warns:
                st.markdown("##### Additional Details")
                cols = st.columns(3)
                if cd:
                    cols[0].metric("Calculated Dose (mg)", f"{cd:.1f}")
                if rd:
                    cols[1].metric("Rounded Dose (mg)", f"{rd}")
                if warns:
                    with cols[2]:
                        st.markdown("<div class='card'><b>Warnings</b><br>" + "<br>".join(warns) + "</div>", unsafe_allow_html=True)

with alt_tab:
    st.markdown('<div class="section-title">Safer or equivalent alternative medications</div>', unsafe_allow_html=True)
    drug_alt = st.text_input("Drug name", value="aspirin", placeholder="e.g., ibuprofen", key="alt_drug_name")
    if st.button("🔁 Suggest Alternatives", use_container_width=True, key="alt_btn"):
        with st.spinner("Finding alternatives by category…"):
            res = api.alternative_suggestions(drug_alt)
        if res:
            alts = res.get('alternatives', [])
            cat = res.get('category', 'Unknown')
            if alts:
                st.markdown(f"**Category:** {cat}")
                st.markdown('<div class="tags">' + ''.join([f"<span>{a}</span>" for a in alts]) + '</div>', unsafe_allow_html=True)
            else:
                st.info(res.get('reason', 'No alternatives found'))

st.markdown("---")
st.caption("Disclaimer: This tool provides informational support and does not replace professional medical advice. Always consult a licensed clinician.")
