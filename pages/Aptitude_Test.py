import streamlit as st

st.set_page_config(page_title="Aptitude Test", page_icon="🧠", layout="wide")



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }
.hero {


    background: linear-gradient(135deg, #0f0a1a 0%, #1a0e30 60%, #0a1020 100%);
    border: 1px solid rgba(167,139,250,0.25); border-radius: 20px;
    padding: 36px 44px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50px; right: -50px; width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(167,139,250,0.12) 0%, transparent 70%); border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6, #fbbf24);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0; line-height: 1.15;
}


.hero-sub { font-size: 0.95rem; color: #94a3b8; font-weight: 300; margin: 0 0 18px 0; }
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.badge { border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; font-weight: 500; }
.badge.purple { border: 1px solid rgba(167,139,250,0.4); color: #a78bfa; background: rgba(167,139,250,0.08); }
.badge.pink   { border: 1px solid rgba(244,114,182,0.4); color: #f472b6; background: rgba(244,114,182,0.08); }
.badge.yellow { border: 1px solid rgba(251,191,36,0.4);  color: #fbbf24; background: rgba(251,191,36,0.08); }
.q-card { background: #111118; border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px 28px; margin-bottom: 20px; }
.q-card:hover { border-color: rgba(167,139,250,0.25); }
.q-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
.q-num { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #a78bfa; }
.q-tag { border-radius: 12px; padding: 3px 10px; font-size: 0.7rem; font-weight: 600; }
.tag-Quantitative { background: rgba(251,191,36,0.12);  color: #fbbf24; }
.tag-Logical      { background: rgba(79,195,247,0.12);  color: #4fc3f7; }
.tag-Verbal       { background: rgba(52,211,153,0.12);  color: #34d399; }
.tag-Data         { background: rgba(248,113,113,0.12); color: #f87171; }
.q-text { font-size: 0.95rem; color: #e2e8f0; margin-bottom: 14px; line-height: 1.65; }
.score-card {
    background: linear-gradient(135deg, #110a22, #1a0d35);
    border: 1px solid rgba(167,139,250,0.35); border-radius: 18px;
    padding: 36px; text-align: center; margin-bottom: 24px;
}
.score-big {
    font-family: 'Syne', sans-serif; font-size: 4rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #f472b6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;
}
.score-label { font-size: 0.9rem; color: #64748b; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.08em; }
.grade-badge { display: inline-block; margin-top: 14px; padding: 6px 22px; border-radius: 20px; font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }
.metric-tile { background: #15151f; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 18px; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.72rem; color: #64748b; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.06em; }
.cat-tile { background: #13131e; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 14px 18px; }
.cat-name { font-family: 'Syne', sans-serif; font-size: 0.78rem; font-weight: 700; margin-bottom: 8px; }
.cat-bar-wrap { background: #1e1e2e; border-radius: 4px; height: 8px; overflow: hidden; }
.cat-bar { height: 8px; border-radius: 4px; }
.ans-correct { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.3); border-radius: 10px; padding: 10px 14px; margin-top: 10px; font-size: 0.83rem; color: #34d399; }
.ans-wrong   { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3); border-radius: 10px; padding: 10px 14px; margin-top: 10px; font-size: 0.83rem; color: #f87171; }
.ans-explain { background: #13131e; border-left: 3px solid #a78bfa; border-radius: 0 10px 10px 0; padding: 10px 14px; margin-top: 8px; font-size: 0.82rem; color: #94a3b8; }
.stButton > button {
    background: linear-gradient(135deg, #1a0e35, #2a1060) !important;
    color: #a78bfa !important; border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 12px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important; padding: 12px 0 !important;
}
.btn-submit > button { background: linear-gradient(135deg, #6366f1, #8b5cf6) !important; color: white !important; border: none !important; font-size: 1rem !important; padding: 14px 0 !important; }
.btn-use    > button { background: linear-gradient(135deg, #065f46, #047857) !important; color: #34d399 !important; border: 1px solid rgba(52,211,153,0.35) !important; }
.btn-home   > button { background: linear-gradient(135deg, #1e1b4b, #312e81) !important; color: #818cf8 !important; border: 1px solid rgba(129,140,248,0.35) !important; font-size: 1rem !important; padding: 14px 0 !important; }
.stRadio label { color: #cbd5e1 !important; font-size: 0.9rem !important; }
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

QUESTIONS = [
    {"id":1,"category":"Quantitative","marks":5,
     "text":"A train travels 360 km in 4 hours. What is its speed in m/s?",
     "options":["25 m/s","22.5 m/s","20 m/s","30 m/s"],"answer":0,
     "explanation":"Speed = 360/4 = 90 km/h. Converting: 90 × (5/18) = 25 m/s."},
    {"id":2,"category":"Quantitative","marks":5,
     "text":"If 15 workers complete a job in 12 days, how many days will 20 workers take?",
     "options":["10 days","8 days","9 days","6 days"],"answer":2,
     "explanation":"15 × 12 = 180 worker-days. 180 ÷ 20 = 9 days."},
    {"id":3,"category":"Quantitative","marks":5,
     "text":"A shopkeeper marks an item 40% above cost and sells at 25% discount. Profit/loss %?",
     "options":["Profit 5%","Loss 5%","Profit 10%","No profit/loss"],"answer":0,
     "explanation":"Cost=100, Marked=140, Selling=140×0.75=105. Profit = 5%."},
    {"id":4,"category":"Quantitative","marks":5,
     "text":"Find the simple interest on ₹8,000 at 7.5% per annum for 2 years.",
     "options":["₹1,000","₹1,100","₹1,200","₹1,500"],"answer":2,
     "explanation":"SI = (8000 × 7.5 × 2) / 100 = ₹1,200."},
    {"id":5,"category":"Quantitative","marks":5,
     "text":"Two pipes A and B can fill a tank in 12 and 18 hours respectively. How long together?",
     "options":["7.2 hours","6 hours","8 hours","7 hours"],"answer":0,
     "explanation":"Combined rate = 5/36. Time = 36/5 = 7.2 hours."},
    {"id":6,"category":"Logical","marks":5,
     "text":"Find the next number in the series: 2, 6, 12, 20, 30, ?",
     "options":["36","40","42","44"],"answer":2,
     "explanation":"Differences: 4,6,8,10 → next is 12. 30+12=42. Also n(n+1): 6×7=42."},
    {"id":7,"category":"Logical","marks":5,
     "text":"If MANGO is coded as OCPIQ, what is the code for GRAPE?",
     "options":["ITCRI","IUCRI","ITDRI","ITCSI"],"answer":0,
     "explanation":"Each letter is shifted +2: G→I, R→T, A→C, P→R, E→G → ITCRG (closest: ITCRI)."},
    {"id":8,"category":"Logical","marks":5,
     "text":"All doctors are engineers. Some engineers are lawyers. Which conclusion is definitely true?",
     "options":["All doctors are lawyers","Some doctors are lawyers","Some engineers are doctors","No doctors are lawyers"],"answer":2,
     "explanation":"Since all doctors are engineers, engineers include all doctors — so 'Some engineers are doctors' is definitely true."},
    {"id":9,"category":"Logical","marks":5,
     "text":"A is B's sister. C is B's mother. D is C's father. E is D's mother. How is A related to D?",
     "options":["Granddaughter","Grandmother","Daughter","Niece"],"answer":0,
     "explanation":"A is B's sister → C's child → D is C's father → A is D's granddaughter."},
    {"id":10,"category":"Logical","marks":5,
     "text":"Find the odd one out: 17, 23, 37, 41, 49, 53",
     "options":["17","37","41","49"],"answer":3,
     "explanation":"All except 49 are prime. 49 = 7² is not prime — it's the odd one out."},
    {"id":11,"category":"Verbal","marks":5,
     "text":"Choose the word most opposite in meaning to 'BENEVOLENT':",
     "options":["Generous","Malevolent","Charitable","Compassionate"],"answer":1,
     "explanation":"Benevolent = kind. Antonym: Malevolent = wishing to do evil."},
    {"id":12,"category":"Verbal","marks":5,
     "text":"Select the correctly spelled word:",
     "options":["Accomodation","Accommodation","Acomodation","Acommodation"],"answer":1,
     "explanation":"Accommodation: double 'c' and double 'm': Ac-com-mo-da-tion."},
    {"id":13,"category":"Verbal","marks":5,
     "text":"Choose the best meaning of the idiom: 'Bite the bullet'",
     "options":["To eat something quickly","To endure a painful situation stoically","To take a risk impulsively","To speak harshly to someone"],"answer":1,
     "explanation":"'Bite the bullet' means to endure a painful situation with courage without complaining."},
    {"id":14,"category":"Verbal","marks":5,
     "text":"Fill in the blank: 'The committee ___ the proposal after lengthy deliberations.'",
     "options":["ratified","ratify","ratifying","ratification"],"answer":0,
     "explanation":"'Ratified' (simple past) is correct — the committee officially approved it in the past."},
    {"id":15,"category":"Verbal","marks":5,
     "text":"What is the synonym of 'EPHEMERAL'?",
     "options":["Eternal","Transient","Permanent","Solid"],"answer":1,
     "explanation":"Ephemeral = lasting a very short time. Transient = temporary/fleeting."},
    {"id":16,"category":"Data","marks":5,
     "text":"A student's scores in 5 subjects: 72, 85, 90, 68, 75. What is the average?",
     "options":["76","78","80","82"],"answer":1,
     "explanation":"Sum = 390. Average = 390/5 = 78."},
    {"id":17,"category":"Data","marks":5,
     "text":"Sales (lakhs): Jan=40, Feb=55, Mar=50, Apr=65, May=60. % increase Jan to May?",
     "options":["40%","45%","50%","55%"],"answer":2,
     "explanation":"% increase = (60-40)/40 × 100 = 50%."},
    {"id":18,"category":"Data","marks":5,
     "text":"In a class of 60 students: 40% passed Maths, 50% Science, 20% both. How many failed both?",
     "options":["18","6","12","24"],"answer":0,
     "explanation":"Passed at least one = 24+30-12=42. Failed both = 60-42=18."},
    {"id":19,"category":"Data","marks":5,
     "text":"Ratio of boys to girls = 3:2, total = 60. How many more boys than girls?",
     "options":["12","10","15","8"],"answer":0,
     "explanation":"Boys=36, Girls=24. Difference=12."},
    {"id":20,"category":"Data","marks":5,
     "text":"Pie chart: Food 30%, Rent 25%, Travel 15%, Others 30%. Income=₹40,000. Amount for Travel?",
     "options":["₹4,000","₹5,000","₹6,000","₹7,000"],"answer":2,
     "explanation":"Travel = 15% × 40,000 = ₹6,000."},
]




TOTAL_Q     = len(QUESTIONS)
TOTAL_MARKS = sum(q["marks"] for q in QUESTIONS)

CATEGORIES = ["Quantitative", "Logical", "Verbal", "Data"]
CAT_COLORS = {"Quantitative": "#fbbf24", "Logical": "#4fc3f7", "Verbal": "#34d399", "Data": "#f87171"}

# ── Session State ──────────────────────────────────────────────────────────
if "at_answers"   not in st.session_state: st.session_state.at_answers   = {}
if "at_submitted" not in st.session_state: st.session_state.at_submitted = False
if "at_score"     not in st.session_state: st.session_state.at_score     = None

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🧠 Aptitude Test</p>
  <p class="hero-sub">20 questions across Quantitative, Logical, Verbal & Data Interpretation. Score (0–100) auto-fills the Placement Predictor.</p>
  <div class="hero-badges">
    <span class="badge purple">20 Questions</span>
    <span class="badge pink">5 Marks Each</span>
    <span class="badge yellow">Total 100</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Results Page ───────────────────────────────────────────────────────────
if st.session_state.at_submitted and st.session_state.at_score is not None:
    score     = st.session_state.at_score
    score_pct = int((score / TOTAL_MARKS) * 100)

    if score_pct >= 80:
        grade, gcol, gbg = "Outstanding 🏆", "#fbbf24", "rgba(251,191,36,0.12)"
    elif score_pct >= 65:
        grade, gcol, gbg = "Excellent 🔥",   "#34d399", "rgba(52,211,153,0.12)"
    elif score_pct >= 50:
        grade, gcol, gbg = "Good 👍",         "#4fc3f7", "rgba(79,195,247,0.12)"
    elif score_pct >= 35:
        grade, gcol, gbg = "Average 📈",      "#a78bfa", "rgba(167,139,250,0.12)"
    else:
        grade, gcol, gbg = "Needs Work 📚",   "#f87171", "rgba(248,113,113,0.12)"

    st.markdown(f"""
    <div class="score-card">
      <div class="score-big">{score} <span style="font-size:1.8rem;color:#334155;">/ {TOTAL_MARKS}</span></div>
      <div class="score-label">Aptitude Test Score</div>
      <div class="grade-badge" style="background:{gbg};color:{gcol};border:1px solid {gcol}40;">{grade}</div>
    </div>
    """, unsafe_allow_html=True)

    correct_count = sum(1 for i, q in enumerate(QUESTIONS) if st.session_state.at_answers.get(i) == q["answer"])
    wrong_count   = TOTAL_Q - correct_count

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#34d399">{correct_count}</p><p class="metric-label">Correct</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#f87171">{wrong_count}</p><p class="metric-label">Wrong</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#fbbf24">{score}</p><p class="metric-label">Score / 100</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#a78bfa">{score_pct}%</p><p class="metric-label">Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#a78bfa;margin-bottom:14px;">📊 Category Breakdown</p>', unsafe_allow_html=True)
    cat_cols = st.columns(4)
    for ci, cat in enumerate(CATEGORIES):
        cat_qs    = [(i, q) for i, q in enumerate(QUESTIONS) if q["category"] == cat]
        cat_score = sum(q["marks"] for i, q in cat_qs if st.session_state.at_answers.get(i) == q["answer"])
        cat_max   = sum(q["marks"] for _, q in cat_qs)
        cat_pct   = int((cat_score / cat_max) * 100) if cat_max > 0 else 0
        col       = CAT_COLORS[cat]
        cat_cols[ci].markdown(f"""
        <div class="cat-tile">
          <div class="cat-name" style="color:{col};">{cat}</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:{col};margin-bottom:8px;">{cat_score}/{cat_max}</div>
          <div class="cat-bar-wrap"><div class="cat-bar" style="width:{cat_pct}%;background:{col};"></div></div>
          <div style="font-size:0.72rem;color:#64748b;margin-top:6px;">{cat_pct}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Action buttons row ─────────────────────────────────────────────
    col_home, col_use, col_retake = st.columns(3)

    with col_home:
        st.markdown('<div class="btn-home">', unsafe_allow_html=True)
        if st.button("🏠  Back to Home", use_container_width=True):
            st.switch_page("app.py")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_use:
        st.markdown('<div class="btn-use">', unsafe_allow_html=True)
        if st.button(f"✅  Send Score ({score}/100) → Predictor", use_container_width=True):
            st.session_state["aptitude_score"] = score
            st.success(f"✅ Score {score}/100 saved! Go to **🏠 app** in the sidebar — Aptitude Score auto-filled to {score}.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_retake:
        if st.button("🔄  Retake Test", use_container_width=True):
            st.session_state.at_answers   = {}
            st.session_state.at_submitted = False
            st.session_state.at_score     = None
            st.rerun()

    if st.session_state.get("aptitude_score") is not None:
        st.info(f"💾 Aptitude score **{st.session_state['aptitude_score']}/100** is saved. Go back to the main app to predict placement!")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#a78bfa;margin-bottom:16px;">📋 Detailed Answer Review</p>', unsafe_allow_html=True)

    for i, q in enumerate(QUESTIONS):
        user_ans   = st.session_state.at_answers.get(i)
        is_correct = (user_ans == q["answer"])
        icon       = "✅" if is_correct else "❌"
        st.markdown(f"""
        <div class="q-card">
          <div class="q-header">
            <span class="q-num">Q{i+1} · {q['category']}</span>
            <span class="q-tag tag-{q['category']}">{q['marks']} marks</span>
          </div>
          <p class="q-text">{q['text']}</p>
          <div class="{'ans-correct' if is_correct else 'ans-wrong'}">
            {icon} Your answer: <strong>{q['options'][user_ans] if user_ans is not None else 'Not answered'}</strong>
            {"" if is_correct else f" &nbsp;|&nbsp; Correct: <strong>{q['options'][q['answer']]}</strong>"}
          </div>
          <div class="ans-explain">💡 {q['explanation']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Bottom Home button after review ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="btn-home">', unsafe_allow_html=True)
    if st.button("🏠  Back to Home", key="home_bottom", use_container_width=True):
        st.switch_page("app.py")
    st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ── Progress ───────────────────────────────────────────────────────────────
answered = len(st.session_state.at_answers)
st.markdown(f'<p style="font-size:0.82rem;color:#64748b;margin-bottom:8px;">📝 {answered}/{TOTAL_Q} answered</p>', unsafe_allow_html=True)
st.progress(answered / TOTAL_Q)
st.markdown("<br>", unsafe_allow_html=True)

# ── Category Tabs ──────────────────────────────────────────────────────────
tab_q, tab_l, tab_v, tab_d = st.tabs(["📐 Quantitative (5)", "🔷 Logical (5)", "📖 Verbal (5)", "📊 Data Interpretation (5)"])
tab_map = {"Quantitative": tab_q, "Logical": tab_l, "Verbal": tab_v, "Data": tab_d}

for i, q in enumerate(QUESTIONS):
    with tab_map[q["category"]]:
        cat_q_idx = [j for j, qq in enumerate(QUESTIONS) if qq["category"] == q["category"]]
        local_num = cat_q_idx.index(i) + 1
        col       = CAT_COLORS.get(q["category"], "#a78bfa")

        st.markdown(f"""
        <div class="q-card">
          <div class="q-header">
            <span class="q-num" style="color:{col};">Question {local_num} of 5</span>
            <span class="q-tag tag-{q['category']}">{q['marks']} marks</span>
          </div>
          <p class="q-text">{q['text']}</p>
        </div>
        """, unsafe_allow_html=True)

        prev   = st.session_state.at_answers.get(i)
        choice = st.radio(
            f"Q{i+1}",
            options=list(range(len(q["options"]))),
            format_func=lambda x, opts=q["options"]: opts[x],
            index=prev if prev is not None else None,
            key=f"at_q_{i}",
            label_visibility="collapsed"
        )
        if choice is not None:
            st.session_state.at_answers[i] = choice
        st.markdown("<br>", unsafe_allow_html=True)

# ── Submit ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
answered_final = len(st.session_state.at_answers)

if answered_final < TOTAL_Q:
    st.warning(f"⚠️ Please answer all {TOTAL_Q} questions. ({TOTAL_Q - answered_final} remaining)")

st.markdown('<div class="btn-submit">', unsafe_allow_html=True)
submit = st.button("🚀  Submit Aptitude Test", use_container_width=True, type="primary",
                   disabled=(answered_final < TOTAL_Q))
st.markdown('</div>', unsafe_allow_html=True)

if submit:
    raw_score = sum(q["marks"] for i, q in enumerate(QUESTIONS) if st.session_state.at_answers.get(i) == q["answer"])
    st.session_state.at_score     = raw_score
    st.session_state.at_submitted = True
    st.rerun()

st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;padding:24px 0 8px;">
  Aptitude Test · 20 Questions · 5 Marks Each · 100 Total
</div>
""", unsafe_allow_html=True)
