import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os


st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6f0; }
.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a0a2e 50%, #0a1628 100%);
    border: 1px solid rgba(138,92,246,0.2); border-radius: 20px;
    padding: 40px 48px; margin-bottom: 32px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -60px; right: -60px; width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(138,92,246,0.15) 0%, transparent 70%); border-radius: 50%;
}


.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0; line-height: 1.15;
}
.hero-sub  { font-size: 1rem; color: #94a3b8; font-weight: 300; margin: 0; }
.hero-badges { display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap; }
.badge { border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; font-weight: 500; }
.badge.purple { border: 1px solid rgba(138,92,246,0.4); color: #a78bfa; background: rgba(138,92,246,0.08); }
.badge.blue   { border: 1px solid rgba(56,189,248,0.4);  color: #38bdf8; background: rgba(56,189,248,0.08); }
.badge.green  { border: 1px solid rgba(52,211,153,0.4);  color: #34d399; background: rgba(52,211,153,0.08); }
.section-card { background: #111118; border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 28px 32px; margin-bottom: 24px; }
.section-label { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #6366f1; margin-bottom: 18px; }
.section-title { font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 700; color: #f1f0f8; margin-bottom: 20px; }
.result-placed     { background: linear-gradient(135deg, #052e16, #0a3d20); border: 1px solid #16a34a; border-radius: 16px; padding: 28px 32px; text-align: center; }
.result-not-placed { background: linear-gradient(135deg, #1c0707, #2d0f0f); border: 1px solid #dc2626; border-radius: 16px; padding: 28px 32px; text-align: center; }
.result-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin: 0 0 6px 0; }
.result-sub   { font-size: 0.9rem; color: #94a3b8; margin: 0; }
.metric-tile  { background: #15151f; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 18px 20px; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.9rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.75rem; color: #64748b; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.06em; }
.fi-row { display:flex; align-items:center; margin-bottom:10px; gap:10px; }
.fi-label { width:160px; font-size:0.82rem; color:#cbd5e1; text-align:right; flex-shrink:0; }
.fi-bar-wrap { flex:1; background:#1e1e2e; border-radius:6px; height:14px; overflow:hidden; }
.fi-bar { height:14px; border-radius:6px; }
.fi-pct { width:48px; font-size:0.8rem; color:#94a3b8; }
.tip-item { background: #13131e; border-left: 3px solid #6366f1; border-radius: 0 10px 10px 0; padding: 12px 16px; margin-bottom: 10px; font-size: 0.88rem; color: #cbd5e1; }
.autofill-notice { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.3); border-radius: 10px; padding: 10px 16px; margin-bottom: 12px; font-size: 0.85rem; color: #34d399; }
.profile-bar-wrap { background: #1e1e2e; border-radius: 8px; height: 18px; overflow: hidden; margin: 10px 0 6px 0; }
.profile-bar { height: 18px; border-radius: 8px; }
.profile-pct-label { font-family: 'Syne', sans-serif; font-size: 1.1rem; font-weight: 700; }
.profile-test-pill { display: inline-block; border-radius: 14px; padding: 3px 12px; font-size: 0.72rem; font-weight: 600; margin: 3px 4px 3px 0; }
.company-card { background: #13131e; border-radius: 12px; padding: 12px 16px; margin-bottom: 10px; display: flex; align-items: center; gap: 12px; }
.company-dot  { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.company-name { font-family: 'Syne', sans-serif; font-size: 0.88rem; font-weight: 700; }
.company-desc { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
.company-badge { margin-left: auto; border-radius: 10px; padding: 2px 10px; font-size: 0.7rem; font-weight: 700; }
.road-item { background: #13131e; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; display: flex; align-items: flex-start; gap: 14px; }
.road-icon  { font-size: 1.2rem; flex-shrink: 0; margin-top: 2px; }
.road-title { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; color: #f1f0f8; }
.road-desc  { font-size: 0.78rem; color: #64748b; margin-top: 3px; line-height: 1.5; }
.road-boost { margin-left: auto; border-radius: 10px; padding: 3px 12px; font-size: 0.72rem; font-weight: 700; flex-shrink: 0; white-space: nowrap; }
.nav-bar {
    display: flex; justify-content: flex-end; align-items: center;
    gap: 10px; margin-bottom: 24px;
}
.nav-label {
    font-size: 0.78rem; color: #475569; margin-right: 8px;
    font-family: 'Syne', sans-serif; letter-spacing: 0.06em; text-transform: uppercase;
}
div[data-testid="stHorizontalBlock"] .nav-btn > button {
    background: #111118 !important;
    color: #a78bfa !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    padding: 8px 10px !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
div[data-testid="stHorizontalBlock"] .nav-btn > button:hover {
    background: rgba(167,139,250,0.12) !important;
    border-color: rgba(167,139,250,0.7) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border: none !important; border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 14px 0 !important; letter-spacing: 0.04em !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
[data-testid="stExpander"] {
    background: #111118 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 16px !important;
    margin-bottom: 24px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #f1f0f8 !important;
    padding: 18px 24px !important;
}
[data-testid="stExpander"] > div > div {
    padding: 0 24px 20px 24px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model        = joblib.load('best_model.pkl')
    encoders     = joblib.load('label_encoders.pkl')
    target_enc   = joblib.load('target_encoder.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    fi = joblib.load('feature_importances.pkl') if os.path.exists('feature_importances.pkl') else None
    return model, encoders, target_enc, feature_cols, fi

model, encoders, target_enc, feature_cols, fi = load_artifacts()

# ── Auto-fill scores from assessment pages ─────────────────────────────────
auto_coding   = st.session_state.get("coding_score",   None)
auto_aptitude = st.session_state.get("aptitude_score", None)
auto_comm     = st.session_state.get("comm_score",     None)

# ── TOP NAVIGATION ─────────────────────────────────────────────────────────
st.markdown('<p style="font-size:0.72rem;color:#475569;font-family:Syne,sans-serif;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">📋 Navigate to Assessments</p>', unsafe_allow_html=True)

nav1, nav2, nav3, nav_spacer = st.columns([1, 1, 1.5, 4.5])
with nav1:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("💻 Coding →", use_container_width=True):
        st.switch_page("pages/Coding_Assessment.py")
    st.markdown('</div>', unsafe_allow_html=True)
with nav2:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("🧠 Aptitude →", use_container_width=True):
        st.switch_page("pages/Aptitude_Test.py")
    st.markdown('</div>', unsafe_allow_html=True)
with nav3:
    st.markdown('<div class="nav-btn">', unsafe_allow_html=True)
    if st.button("🗣 Communication →", use_container_width=True):
        st.switch_page("pages/Communication_Test.py")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🎓 Placement Predictor</p>
  <p class="hero-sub">Fill your profile below — or take the Coding, Aptitude &amp; Communication tests from the sidebar to auto-fill your scores.</p>
  <div class="hero-badges">
    <span class="badge purple">Random Forest Model</span>
    <span class="badge blue">50,000 Student Records</span>
    <span class="badge green">Real Feature Importance</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Profile Completeness Bar ───────────────────────────────────────────────
tests_done  = sum([auto_coding is not None, auto_aptitude is not None, auto_comm is not None])
profile_pct = [0, 34, 67, 100][tests_done]

if   profile_pct == 100: bar_color, bar_msg = "#34d399", "Profile Complete 🎉"
elif profile_pct >= 67:  bar_color, bar_msg = "#38bdf8", "Almost There!"
elif profile_pct >= 34:  bar_color, bar_msg = "#a78bfa", "Good Progress"
else:                    bar_color, bar_msg = "#f87171", "Complete Your Profile"

pill_c = f'<span class="profile-test-pill" style="background:rgba(79,195,247,0.15);color:#4fc3f7;border:1px solid rgba(79,195,247,0.3);">{"✅" if auto_coding   is not None else "⬜"} Coding Test</span>'
pill_a = f'<span class="profile-test-pill" style="background:rgba(167,139,250,0.15);color:#a78bfa;border:1px solid rgba(167,139,250,0.3);">{"✅" if auto_aptitude is not None else "⬜"} Aptitude Test</span>'
pill_m = f'<span class="profile-test-pill" style="background:rgba(52,211,153,0.15);color:#34d399;border:1px solid rgba(52,211,153,0.3);">{"✅" if auto_comm     is not None else "⬜"} Communication Test</span>'

st.markdown(f"""
<div class="section-card" style="padding:20px 28px;margin-bottom:20px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
    <span style="font-family:'Syne',sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#6366f1;">Profile Completeness</span>
    <span class="profile-pct-label" style="color:{bar_color};">{profile_pct}% — {bar_msg}</span>
  </div>
  <div class="profile-bar-wrap">
    <div class="profile-bar" style="width:{profile_pct}%;background:linear-gradient(90deg,{bar_color},{bar_color}bb);"></div>
  </div>
  <div style="margin-top:10px;">{pill_c} {pill_a} {pill_m}</div>
</div>
""", unsafe_allow_html=True)

# ── Auto-fill banners ──────────────────────────────────────────────────────
if auto_coding   is not None:
    st.markdown(f'<div class="autofill-notice">✅ Coding Skills auto-filled from assessment: <strong>{auto_coding}/10</strong></div>', unsafe_allow_html=True)
if auto_aptitude is not None:
    st.markdown(f'<div class="autofill-notice">✅ Aptitude Score auto-filled from test: <strong>{auto_aptitude}/100</strong></div>', unsafe_allow_html=True)
if auto_comm     is not None:
    st.markdown(f'<div class="autofill-notice">✅ Communication Skills auto-filled from test: <strong>{auto_comm}/10</strong></div>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Assessment Pages")
    st.info("Take all 3 tests to reach 100% profile completeness and unlock prediction.")
    if st.button("💻 Go to Coding Assessment", use_container_width=True):
        st.switch_page("pages/Coding_Assessment.py")
    if st.button("🧠 Go to Aptitude Test", use_container_width=True):
        st.switch_page("pages/Aptitude_Test.py")
    if st.button("🗣 Go to Communication Test", use_container_width=True):
        st.switch_page("pages/Communication_Test.py")
    st.markdown("---")
    if auto_coding   is not None: st.success(f"💻 Coding: **{auto_coding}/10**")
    if auto_aptitude is not None: st.success(f"🧠 Aptitude: **{auto_aptitude}/100**")
    if auto_comm     is not None: st.success(f"🗣 Communication: **{auto_comm}/10**")
    st.markdown("---")
    st.markdown(f"**Profile:** `{profile_pct}%` complete")

# ── Feature Importance ─────────────────────────────────────────────────────
if fi is not None:
    with st.expander("🌟 Key Factors for Placement — From Trained Random Forest", expanded=True):
        fi_sorted = fi.sort_values(ascending=False)
        max_val   = fi_sorted.max()
        def fi_color(val):
            if val == max_val:                      return '#f87171'
            elif val >= fi_sorted.quantile(0.75):   return '#4fc3f7'
            elif val >= fi_sorted.quantile(0.5):    return '#a78bfa'
            else:                                   return '#6366f1'
        rows_html = ""
        for feat, val in fi_sorted.items():
            bar_w = int((val / max_val) * 100)
            color = fi_color(val)
            rows_html += f"""
            <div class="fi-row">
              <div class="fi-label">{feat}</div>
              <div class="fi-bar-wrap"><div class="fi-bar" style="width:{bar_w}%;background:{color};"></div></div>
              <div class="fi-pct">{val*100:.2f}%</div>
            </div>"""
        st.markdown(rows_html, unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex;gap:18px;margin-top:14px;flex-wrap:wrap;font-size:0.78rem;">
          <span style="color:#f87171;">● Most Important</span>
          <span style="color:#4fc3f7;">● High Importance</span>
          <span style="color:#a78bfa;">● Moderate</span>
          <span style="color:#6366f1;">● Lower</span>
        </div>""", unsafe_allow_html=True)

# ── Input Form ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">Step 01</p>'
            '<p class="section-title">Academic & Personal Details</p>',
            unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    age         = st.slider("🎂 Age",                        5, 55, 0)
    gender      = st.selectbox("👤 Gender",                  ["Male", "Female"])
    degree      = st.selectbox("🎓 Degree",                  ["B.Tech", "BCA", "MCA", "B.Sc"])
    branch      = st.selectbox("🏫 Branch",                  ["CSE", "ECE", "ME", "Civil", "IT"])
    cgpa        = st.slider("📊 CGPA",                       0.0, 10.0, 0.0, step=0.1, format="%.1f")
with col2:
    internships = st.slider("🏢 Internships",                0, 3, 0)
    projects    = st.slider("🛠 Projects",                   0, 6, 0)
    coding_def  = int(auto_coding)   if auto_coding   is not None else 0
    coding      = st.slider("💻 Coding Skills (0–10)",       0, 10, coding_def)
    comm_def    = int(auto_comm)     if auto_comm     is not None else 0
    comm        = st.slider("🗣 Communication Skills (0–10)", 0, 10, comm_def)
    apt_def     = int(auto_aptitude) if auto_aptitude is not None else 0
    aptitude    = st.slider("🧠 Aptitude Test Score (0–100)", 0, 100, apt_def)
with col3:
    soft        = st.slider("🌟 Soft Skills Rating",         0, 10, 0)
    certs       = st.slider("📜 Certifications",             0, 3, 0)
    backlogs    = st.slider("⚠️ Backlogs",                  0, 3, 0)

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict — gated behind 100% profile ───────────────────────────────────
if profile_pct < 100:
    remaining  = 3 - tests_done
    tests_left = []
    if auto_coding   is None: tests_left.append("💻 Coding Test")
    if auto_aptitude is None: tests_left.append("🧠 Aptitude Test")
    if auto_comm     is None: tests_left.append("🗣 Communication Test")

    st.markdown(f"""
    <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.35);
         border-radius:14px;padding:20px 24px;margin-bottom:16px;">
      <p style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;color:#f87171;margin:0 0 6px 0;">
        🔒 Complete your profile to unlock placement prediction
      </p>
      <p style="font-size:0.87rem;color:#94a3b8;margin:0 0 12px 0;">
        You are at <strong style="color:#f87171;">{profile_pct}%</strong>.
        Take the remaining {remaining} test{"s" if remaining > 1 else ""} from the sidebar or use the buttons above.
      </p>
      <div>{"".join(
          f'<span style="display:inline-block;background:rgba(248,113,113,0.12);color:#f87171;'
          f'border:1px solid rgba(248,113,113,0.3);border-radius:10px;padding:3px 14px;'
          f'font-size:0.78rem;font-weight:600;margin:3px 6px 3px 0;">{t}</span>'
          for t in tests_left
      )}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Quick Jump →**")
    jc1, jc2, jc3 = st.columns(3)
    with jc1:
        if auto_coding is None:
            if st.button("💻 Take Coding Test", use_container_width=True):
                st.switch_page("pages/Coding_Assessment.py")
    with jc2:
        if auto_aptitude is None:
            if st.button("🧠 Take Aptitude Test", use_container_width=True):
                st.switch_page("pages/Aptitude_Test.py")
    with jc3:
        if auto_comm is None:
            if st.button("🗣 Take Communication Test", use_container_width=True):
                st.switch_page("pages/Communication_Test.py")

    st.button("🔒  Complete Profile to Predict", use_container_width=True,
              type="primary", disabled=True)
    st.stop()

predict_btn = st.button("🔍  Predict My Placement", use_container_width=True, type="primary")

# ── Run prediction ─────────────────────────────────────────────────────────
if predict_btn:
    if comm < 5:
        st.markdown("""
        <div class="result-not-placed">
          <p class="result-title" style="color:#f87171;">❌ NOT PLACED</p>
          <p class="result-sub">Communication Skills must be at least 5 to be eligible.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="tip-item">🗣 Take the Communication Skills Test from the sidebar to see exactly where you stand and get personalised tips.</div>', unsafe_allow_html=True)
        st.stop()

    if backlogs >= 2:
        st.markdown("""
        <div class="result-not-placed">
          <p class="result-title" style="color:#f87171;">❌ NOT PLACED</p>
          <p class="result-sub">Students with 2 or more backlogs are not eligible.</p>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="tip-item">📚 Clear your backlogs — most companies enforce a strict 0–1 backlog policy.</div>', unsafe_allow_html=True)
        st.stop()

    try:
        gender_enc = encoders['Gender'].transform([gender])[0]
        degree_enc = encoders['Degree'].transform([degree])[0]
        branch_enc = encoders['Branch'].transform([branch])[0]
    except Exception as e:
        st.warning(f"Encoding issue: {e}")
        gender_enc, degree_enc, branch_enc = 0, 0, 0

    row = pd.DataFrame(
        [[age, gender_enc, degree_enc, branch_enc, cgpa,
          internships, projects, coding, comm, aptitude,
          soft, certs, backlogs]],
        columns=feature_cols
    )
    pred  = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    label = target_enc.inverse_transform([pred])[0]

    classes        = list(model.classes_)
    placed_idx     = classes.index(target_enc.transform(["Placed"])[0])
    not_placed_idx = 1 - placed_idx
    placed_pct     = round(proba[placed_idx] * 100, 1)
    not_placed_pct = round(proba[not_placed_idx] * 100, 1)

    st.markdown("<br>", unsafe_allow_html=True)
    if label == "Placed":
        st.markdown(f"""
        <div class="result-placed">
          <p class="result-title" style="color:#4ade80;">✅ PLACED!</p>
          <p class="result-sub">Your profile meets placement criteria. Confidence: <strong style="color:#4ade80">{placed_pct}%</strong></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-not-placed">
          <p class="result-title" style="color:#f87171;">❌ NOT PLACED</p>
          <p class="result-sub">Your profile needs improvement. Confidence: <strong style="color:#f87171">{not_placed_pct}%</strong></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#4ade80">{placed_pct}%</p><p class="metric-label">Placed Chance</p></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#f87171">{not_placed_pct}%</p><p class="metric-label">Not Placed</p></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#a78bfa">{cgpa}</p><p class="metric-label">Your CGPA</p></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#38bdf8">{aptitude}</p><p class="metric-label">Aptitude Score</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(placed_pct / 100, text=f"Placement probability: {placed_pct}%")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<p class="section-label">🌟 Feature Importance</p>', unsafe_allow_html=True)
        if fi is not None:
            fi_plot    = fi.sort_values(ascending=True)
            bar_colors = ['#f87171' if v == fi_plot.max() else
                          '#4fc3f7' if v >= fi_plot.quantile(0.75) else
                          '#6366f1' for v in fi_plot.values]
            fig1, ax1  = plt.subplots(figsize=(4.5, 4))
            fig1.patch.set_facecolor('#111118'); ax1.set_facecolor('#111118')
            bars = ax1.barh(fi_plot.index, fi_plot.values * 100, color=bar_colors, height=0.55, edgecolor='none')
            ax1.set_xlabel('Importance (%)', color='#64748b', fontsize=8)
            ax1.tick_params(colors='#94a3b8', labelsize=7)
            for spine in ax1.spines.values(): spine.set_visible(False)
            ax1.set_xlim(0, fi_plot.max() * 100 * 1.3)
            for bar, val in zip(bars, fi_plot.values):
                ax1.text(val * 100 + 0.2, bar.get_y() + bar.get_height()/2,
                         f'{val*100:.1f}%', va='center', color='#94a3b8', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True)
            plt.close()

    with c2:
        st.markdown('<p class="section-label">Placement Probability</p>', unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        fig2.patch.set_facecolor('#111118'); ax2.set_facecolor('#111118')
        wedges, texts, autotexts = ax2.pie(
            [placed_pct, not_placed_pct], explode=(0.04, 0),
            colors=['#4ade80', '#f87171'], autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor': '#111118', 'linewidth': 2},
            textprops={'color': '#e2e8f0', 'fontsize': 10}
        )
        for at in autotexts: at.set_fontsize(10); at.set_color('#0a0a0f'); at.set_fontweight('bold')
        ax2.legend(
            handles=[mpatches.Patch(color='#4ade80', label=f'Placed ({placed_pct}%)'),
                     mpatches.Patch(color='#f87171', label=f'Not Placed ({not_placed_pct}%)')],
            loc='lower center', framealpha=0, labelcolor='#94a3b8', fontsize=8
        )
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    with c3:
        st.markdown('<p class="section-label">🕸 Skill Radar</p>', unsafe_allow_html=True)
        radar_labels = ['CGPA', 'Coding', 'Comm.', 'Aptitude', 'Soft Skills', 'Projects']
        radar_values = [
            cgpa, coding, comm,
            round(aptitude / 10, 1), soft,
            round(min(projects / 6 * 10, 10), 1),
        ]
        N      = len(radar_labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        vals   = radar_values + [radar_values[0]]
        angs   = angles + angles[:1]
        fig3, ax3 = plt.subplots(figsize=(4.5, 4), subplot_kw=dict(polar=True))
        fig3.patch.set_facecolor('#111118'); ax3.set_facecolor('#111118')
        ax3.set_ylim(0, 10)
        ax3.set_yticks([2, 4, 6, 8, 10])
        ax3.set_yticklabels(['2','4','6','8','10'], color='#334155', fontsize=6)
        ax3.set_xticks(angles)
        ax3.set_xticklabels(radar_labels, color='#94a3b8', fontsize=8)
        ax3.grid(color='#1e1e2e', linewidth=0.8)
        ax3.spines['polar'].set_color('#1e1e2e')
        ax3.plot(angs, vals, color='#a78bfa', linewidth=2)
        ax3.fill(angs, vals, color='#a78bfa', alpha=0.18)
        for angle, val in zip(angles, radar_values):
            dot_col = '#f87171' if val < 4 else '#fbbf24' if val < 7 else '#34d399'
            ax3.plot(angle, val, 'o', color=dot_col, markersize=6, zorder=5)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Company Matcher + Improvement Roadmap ──────────────────────────
    comp_col, road_col = st.columns(2)
    with comp_col:
        st.markdown('<p class="section-label">🏢 Company Matcher</p>', unsafe_allow_html=True)
        companies = [
            {"name": "Product Companies",  "desc": "Google, Microsoft, Amazon, Flipkart",
             "color": "#f87171",
             "eligible": cgpa >= 8.0 and coding >= 8 and backlogs == 0 and comm >= 7,
             "req": "CGPA ≥ 8.0 · Coding ≥ 8 · No backlogs · Comm ≥ 7"},
            {"name": "Mid-tier Tech",      "desc": "Zoho, Freshworks, Hexaware, Mphasis",
             "color": "#38bdf8",
             "eligible": cgpa >= 7.0 and coding >= 6 and backlogs <= 1 and comm >= 6,
             "req": "CGPA ≥ 7.0 · Coding ≥ 6 · Backlogs ≤ 1 · Comm ≥ 6"},
            {"name": "Service Companies",  "desc": "TCS, Infosys, Wipro, Cognizant",
             "color": "#a78bfa",
             "eligible": cgpa >= 6.0 and backlogs <= 1 and comm >= 5,
             "req": "CGPA ≥ 6.0 · Backlogs ≤ 1 · Comm ≥ 5"},
            {"name": "Startups",           "desc": "Early/growth-stage, project-driven",
             "color": "#34d399",
             "eligible": coding >= 5 and projects >= 2 and comm >= 5,
             "req": "Coding ≥ 5 · Projects ≥ 2 · Comm ≥ 5"},
            {"name": "Consulting / Analytics", "desc": "Deloitte, EY, KPMG, Mu Sigma",
             "color": "#fbbf24",
             "eligible": cgpa >= 6.5 and aptitude >= 60 and comm >= 6,
             "req": "CGPA ≥ 6.5 · Aptitude ≥ 60 · Comm ≥ 6"},
        ]
        any_eligible = False
        for c in companies:
            s_bg   = "rgba(52,211,153,0.12)"  if c["eligible"] else "rgba(248,113,113,0.08)"
            s_col  = "#34d399"                 if c["eligible"] else "#f87171"
            s_text = "✅ Eligible"             if c["eligible"] else "❌ Not yet"
            border = c["color"]                if c["eligible"] else "rgba(255,255,255,0.05)"
            if c["eligible"]: any_eligible = True
            st.markdown(f"""
            <div class="company-card" style="border:1px solid {border};">
              <div class="company-dot" style="background:{c['color']};"></div>
              <div style="flex:1;">
                <div class="company-name" style="color:{c['color']};">{c['name']}</div>
                <div class="company-desc">{c['desc']}</div>
                <div style="font-size:0.68rem;color:#475569;margin-top:3px;">{c['req']}</div>
              </div>
              <div class="company-badge" style="background:{s_bg};color:{s_col};">{s_text}</div>
            </div>
            """, unsafe_allow_html=True)
        if not any_eligible:
            st.markdown('<div class="tip-item" style="border-color:#f87171;">⚠️ No tier matched yet — focus on CGPA, communication, and coding first.</div>', unsafe_allow_html=True)

    with road_col:
        st.markdown('<p class="section-label">🗺 Improvement Roadmap</p>', unsafe_allow_html=True)
        roadmap = []
        if backlogs >= 1:
            roadmap.append({"icon":"🚨","priority":"critical","title":"Clear your backlogs",
                "desc":"Most companies auto-reject students with active backlogs.", "boost":"+15%"})
        if cgpa < 6.5:
            roadmap.append({"icon":"📚","priority":"critical","title":"Improve CGPA above 6.5",
                "desc":"CGPA gates you out of most drives.", "boost":"+12%"})
        if comm < 6:
            roadmap.append({"icon":"🗣","priority":"critical","title":"Strengthen communication skills",
                "desc":"Communication carries the highest model weight.", "boost":"+10%"})
        if coding < 6:
            roadmap.append({"icon":"💻","priority":"important","title":"Level up coding skills",
                "desc":"Solve 50+ LeetCode Easy/Medium problems.", "boost":"+8%"})
        if internships == 0:
            roadmap.append({"icon":"🏢","priority":"important","title":"Get at least one internship",
                "desc":"Apply on Internshala, LinkedIn, and company portals.", "boost":"+7%"})
        if aptitude < 60:
            roadmap.append({"icon":"🧠","priority":"important","title":"Practice aptitude daily",
                "desc":"Spend 20 min/day on IndiaBix or PrepInsta.", "boost":"+6%"})
        if projects < 2:
            roadmap.append({"icon":"🛠","priority":"good","title":"Build 2–3 strong projects",
                "desc":"Host on GitHub with a good README.", "boost":"+5%"})
        if certs == 0:
            roadmap.append({"icon":"📜","priority":"good","title":"Earn 1–2 certifications",
                "desc":"Free options: Google (Coursera), NPTEL, AWS.", "boost":"+4%"})
        if soft < 6:
            roadmap.append({"icon":"🌟","priority":"good","title":"Work on soft skills",
                "desc":"Participate in hackathons and team projects.", "boost":"+3%"})

        priority_styles = {
            "critical":  ("#f87171", "rgba(248,113,113,0.12)"),
            "important": ("#fbbf24", "rgba(251,191,36,0.10)"),
            "good":      ("#34d399", "rgba(52,211,153,0.10)"),
        }
        if not roadmap:
            st.markdown("""
            <div class="company-card" style="border:1px solid rgba(52,211,153,0.3);background:rgba(52,211,153,0.06);">
              <div style="font-size:1.4rem;">🎯</div>
              <div>
                <div class="company-name" style="color:#34d399;">Your profile looks strong!</div>
                <div class="company-desc">Focus on interview prep, mock GDs, and applying early.</div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            for item in roadmap[:4]:
                pcol, pbg = priority_styles[item["priority"]]
                st.markdown(f"""
                <div class="road-item" style="border-left:3px solid {pcol};">
                  <div class="road-icon">{item['icon']}</div>
                  <div style="flex:1;">
                    <div class="road-title">{item['title']}</div>
                    <div class="road-desc">{item['desc']}</div>
                  </div>
                  <div class="road-boost" style="background:{pbg};color:{pcol};">{item['boost']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Quick tips ─────────────────────────────────────────────────────
    tips = []
    if cgpa < 6.5:       tips.append("📚 Improve your CGPA — aim for at least 7.0")
    if coding < 6:       tips.append("💻 Strengthen coding skills — practice DSA on LeetCode / HackerRank")
    if comm < 7:         tips.append("🗣 Work on communication — take the Communication Skills Test")
    if internships == 0: tips.append("🏢 Complete at least one internship before placement season")
    if certs == 0:       tips.append("📜 Earn certifications on Coursera, NPTEL, or Google")
    if projects < 2:     tips.append("🛠 Build 2–3 strong projects and host them on GitHub")

    if tips:
        st.markdown('<p class="section-label">💡 Quick Suggestions</p>', unsafe_allow_html=True)
        for tip in tips:
            st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;padding:16px 0;">
  ML Mini Project · Placement Prediction System · Built with Streamlit &amp; scikit-learn
</div>
""", unsafe_allow_html=True)
