import streamlit as st

st.set_page_config(page_title="Communication Test", page_icon="🗣", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6f0; }


.hero {
    background: linear-gradient(135deg, #0f0f1a 0%, #0a1a2e 100%);
    border: 1px solid rgba(52,211,153,0.2); border-radius: 20px;
    padding: 36px 44px; margin-bottom: 28px;
}

.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(90deg, #34d399, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}


.hero-sub { font-size: 0.95rem; color: #94a3b8; margin: 0; }
.q-card {
    background: #111118; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 22px 26px; margin-bottom: 18px;
}
.q-num { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
         text-transform: uppercase; color: #6366f1; margin-bottom: 8px; }
.q-text { font-size: 0.95rem; color: #e2e8f0; font-weight: 500; margin-bottom: 14px; line-height: 1.55; }
.result-card {
    background: linear-gradient(135deg, #0a1f14, #052e1c);
    border: 1px solid #16a34a; border-radius: 18px; padding: 32px 36px; margin-top: 20px;
}


.result-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800;
                color: #4ade80; margin: 0 0 4px 0; }
.result-grade { font-size: 1rem; color: #86efac; margin: 0 0 18px 0; }
.skill-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.skill-label { width: 180px; font-size: 0.82rem; color: #cbd5e1; flex-shrink: 0; }
.skill-bar-wrap { flex: 1; background: #1e1e2e; border-radius: 6px; height: 12px; overflow: hidden; }
.skill-bar { height: 12px; border-radius: 6px; }
.skill-pct { width: 42px; font-size: 0.8rem; color: #94a3b8; text-align: right; }
.remark-box { background: rgba(52,211,153,0.07); border-left: 3px solid #34d399;
              border-radius: 0 10px 10px 0; padding: 14px 18px; margin: 16px 0;
              font-size: 0.88rem; color: #a7f3d0; line-height: 1.6; }
.prof-footer { border-top: 1px solid rgba(255,255,255,0.06); margin-top: 20px; padding-top: 16px;
               font-size: 0.78rem; color: #475569; text-align: right; }
.metric-tile { background: #15151f; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 18px; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.72rem; color: #64748b; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.06em; }
.score-card {
    background: linear-gradient(135deg, #0a1f14, #052e1c);
    border: 1px solid rgba(52,211,153,0.35); border-radius: 18px;
    padding: 36px; text-align: center; margin-bottom: 24px;
}
.score-big {
    font-family: 'Syne', sans-serif; font-size: 4rem; font-weight: 800;
    background: linear-gradient(90deg, #34d399, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;
}
.score-label { font-size: 0.9rem; color: #64748b; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.08em; }
.grade-badge { display: inline-block; margin-top: 14px; padding: 6px 22px; border-radius: 20px;
               font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }
.btn-submit > button { background: linear-gradient(135deg, #059669, #0891b2) !important; color: white !important; border: none !important; border-radius: 12px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 1rem !important; padding: 14px 0 !important; }
.btn-use > button { background: linear-gradient(135deg, #065f46, #047857) !important; color: #34d399 !important; border: 1px solid rgba(52,211,153,0.35) !important; border-radius: 12px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important; padding: 12px 0 !important; }
.btn-home > button { background: linear-gradient(135deg, #1e3a5f, #1e40af) !important; color: #93c5fd !important; border: 1px solid rgba(147,197,253,0.35) !important; border-radius: 12px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important; padding: 12px 0 !important; }
.stButton > button { background: linear-gradient(135deg, #1a1a2e, #2a1060) !important; color: #a78bfa !important; border: 1px solid rgba(167,139,250,0.3) !important; border-radius: 12px !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.95rem !important; padding: 12px 0 !important; }
.stRadio label { color: #cbd5e1 !important; font-size: 0.9rem !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 900px; }
hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">🗣 Communication Skills Assessment</p>
  <p class="hero-sub">10 professional scenario-based questions · Evaluated by department standard · Score auto-fills Placement Predictor</p>
</div>
""", unsafe_allow_html=True)

# ── Questions ──────────────────────────────────────────────────────────────
questions = [
    {
        "text": "You have been asked to present your semester project to a panel of faculty members. Just before your turn, you realise one of your slides contains an error. What is the most appropriate course of action?",
        "options": ["Skip that slide entirely without mentioning it", "Acknowledge the error at the start, correct it verbally, and proceed confidently", "Present the slide as-is and hope the panel does not notice", "Ask to postpone your presentation to fix the slide"],
        "answer": 1
    },
    {
        "text": "During a group discussion in class, a fellow student interrupts you repeatedly while you are making a point. How should you handle this situation professionally?",
        "options": ["Raise your voice to continue speaking over them", "Stop talking and let them finish, then politely reclaim your point", "Complain to the professor immediately", "Abandon your point and remain silent for the rest of the discussion"],
        "answer": 1
    },
    {
        "text": "You receive an email from your professor asking for a project update. You have not made much progress. What is the most professional response?",
        "options": ["Ignore the email until you have something substantial to report", "Reply with a vague message saying 'It is going well'", "Acknowledge the email, honestly state your current status, and outline your plan going forward", "Ask a classmate to update on your behalf"],
        "answer": 2
    },
    {
        "text": "During a technical team presentation, your colleague provides incorrect data while presenting. It is now your turn to speak. What should you do?",
        "options": ["Interrupt them immediately to correct the error in front of everyone", "Ignore the mistake and proceed with your section", "Discreetly clarify the correct data point when it is relevant to your section", "Announce the error loudly after the presentation ends"],
        "answer": 2
    },
    {
        "text": "You are attending a formal placement interview. The interviewer asks a question you do not fully understand. Which response best demonstrates strong communication skills?",
        "options": ["Answer based on your best guess without seeking clarification", "Politely say 'I am not sure' and move on", "Pause, politely ask the interviewer to clarify, then answer thoughtfully", "Ask the interviewer to repeat every question regardless of clarity"],
        "answer": 2
    },
    {
        "text": "Your team is in conflict about the direction of a project. As a member (not the leader), what is the most constructive communication approach?",
        "options": ["Argue strongly for your own idea until others agree", "Remain completely silent to avoid conflict", "Listen to all perspectives, summarise the points of agreement, and propose a collaborative way forward", "Privately side with the majority and avoid voicing your opinion"],
        "answer": 2
    },
    {
        "text": "You must deliver bad news to your project team — the client has rejected your proposal. How do you communicate this?",
        "options": ["Blame the client and tell the team the rejection was unfair", "Inform the team clearly and factually, acknowledge the setback, and immediately shift focus to next steps", "Delay informing the team hoping the situation resolves itself", "Tell only the team leader and let them handle the communication"],
        "answer": 1
    },
    {
        "text": "During a formal seminar, an audience member challenges your research findings aggressively. What is the ideal response?",
        "options": ["Challenge them back with equal aggression to assert your position", "Apologise and concede all your points to avoid confrontation", "Remain calm, thank them for the question, present your evidence clearly, and acknowledge any genuine limitations", "Ignore the question and move on to the next slide"],
        "answer": 2
    },
    {
        "text": "You are writing a formal email to a company's HR department requesting an internship opportunity. Which of the following best reflects professional written communication?",
        "options": ["A casual, brief message saying 'Hi, I want an internship, please let me know'", "A long email listing every achievement without any specific relevance to the role", "A concise, formally structured email with a proper salutation, clear intent, relevant qualifications, and a polite closing", "Copying a template from the internet without editing it for the specific company"],
        "answer": 2
    },
    {
        "text": "After completing your internship, your supervisor asks you to give feedback on the company's work processes. You have some genuine concerns. How do you communicate this?",
        "options": ["Praise everything to maintain a positive relationship", "List all criticisms bluntly without any context or suggestions", "Provide balanced, specific, and respectful feedback — acknowledge strengths and frame concerns as constructive suggestions", "Refuse to give feedback to avoid any potential negative impression"],
        "answer": 2
    },
]

# ── Session State ──────────────────────────────────────────────────────────
if "ct_answers"   not in st.session_state: st.session_state.ct_answers   = {}
if "ct_submitted" not in st.session_state: st.session_state.ct_submitted = False
if "ct_score"     not in st.session_state: st.session_state.ct_score     = None

# ── Results Page ───────────────────────────────────────────────────────────
if st.session_state.ct_submitted and st.session_state.ct_score is not None:
    score_10 = st.session_state.ct_score
    pct      = score_10 * 10

    if score_10 >= 9:
        grade, label, bar_color, gcol, gbg = "A+", "Outstanding",       "#34d399", "#34d399", "rgba(52,211,153,0.12)"
    elif score_10 >= 8:
        grade, label, bar_color, gcol, gbg = "A",  "Excellent",         "#4ade80", "#4ade80", "rgba(74,222,128,0.12)"
    elif score_10 >= 7:
        grade, label, bar_color, gcol, gbg = "B+", "Very Good",         "#38bdf8", "#38bdf8", "rgba(56,189,248,0.12)"
    elif score_10 >= 6:
        grade, label, bar_color, gcol, gbg = "B",  "Good",              "#a78bfa", "#a78bfa", "rgba(167,139,250,0.12)"
    elif score_10 >= 5:
        grade, label, bar_color, gcol, gbg = "C",  "Satisfactory",      "#fbbf24", "#fbbf24", "rgba(251,191,36,0.12)"
    else:
        grade, label, bar_color, gcol, gbg = "D",  "Needs Improvement", "#f87171", "#f87171", "rgba(248,113,113,0.12)"

    # ── Score card ─────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="score-card">
      <div class="score-big">{score_10} <span style="font-size:1.8rem;color:#334155;">/ 10</span></div>
      <div class="score-label">Communication Skills Score</div>
      <div class="grade-badge" style="background:{gbg};color:{gcol};border:1px solid {gcol}40;">{grade} — {label}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric tiles ───────────────────────────────────────────────────
    correct_count = score_10
    wrong_count   = 10 - score_10
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#34d399">{correct_count}</p><p class="metric-label">Correct</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#f87171">{wrong_count}</p><p class="metric-label">Wrong</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#fbbf24">{score_10}</p><p class="metric-label">Score / 10</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#a78bfa">{pct}%</p><p class="metric-label">Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Progress bar ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin:10px 0 20px;">
      <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#64748b;margin-bottom:5px;">
        <span>Communication Proficiency</span><span>{pct}%</span>
      </div>
      <div style="background:#1e1e2e;border-radius:8px;height:16px;overflow:hidden;">
        <div style="width:{pct}%;height:16px;background:{bar_color};border-radius:8px;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Skill breakdown — rendered SEPARATELY to avoid escaping ────────
    answers_saved = st.session_state.ct_answers
    skills = {
        "Professional Conduct":    min(10, max(0, score_10 + (1 if answers_saved.get(0) == questions[0]["answer"] else -1))),
        "Conflict Resolution":     min(10, max(0, score_10 + (1 if answers_saved.get(1) == questions[1]["answer"] else -1))),
        "Written Communication":   min(10, max(0, score_10 + (1 if answers_saved.get(8) == questions[8]["answer"] else -1))),
        "Active Listening":        min(10, max(0, score_10 + (1 if answers_saved.get(4) == questions[4]["answer"] else -1))),
        "Feedback & Adaptability": min(10, max(0, score_10 + (1 if answers_saved.get(9) == questions[9]["answer"] else -1))),
    }

    st.markdown('<p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:#6366f1;margin-bottom:12px;">Skill Breakdown</p>', unsafe_allow_html=True)
    for sk, sv in skills.items():
        w   = int(sv * 10)
        col = "#34d399" if sv >= 8 else "#38bdf8" if sv >= 6 else "#fbbf24" if sv >= 4 else "#f87171"
        st.markdown(f"""
        <div class="skill-row">
          <div class="skill-label">{sk}</div>
          <div class="skill-bar-wrap"><div class="skill-bar" style="width:{w}%;background:{col};"></div></div>
          <div class="skill-pct">{sv}/10</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Professor remarks ──────────────────────────────────────────────
    if score_10 >= 9:
        remark = ("This student demonstrates an exceptional command of professional communication. "
                  "The responses reflect mature judgment, situational awareness, and the ability to navigate "
                  "complex interpersonal dynamics with confidence and grace. This performance is commendable "
                  "and indicative of a student ready for high-responsibility professional environments. "
                  "It is a pleasure to evaluate such a well-rounded communicator.")
        quote  = '"Communication works for those who work at it." — John Powell'
    elif score_10 >= 7:
        remark = ("The student has demonstrated a solid understanding of professional communication norms "
                  "and etiquette. Responses reflect good situational reasoning and an awareness of workplace "
                  "dynamics. With continued practice in written correspondence and conflict resolution, "
                  "this student is well-positioned for campus placements. A commendable effort overall.")
        quote  = '"The art of communication is the language of leadership." — James Humes'
    elif score_10 >= 5:
        remark = ("The student shows a foundational understanding of communication principles. "
                  "Several responses reflect sound professional instincts, though there is room to develop "
                  "deeper sensitivity to formal contexts and structured written communication. "
                  "Consistent practice and exposure to professional scenarios will strengthen this skill set significantly.")
        quote  = '"To effectively communicate, we must realise that we are all different." — Tony Robbins'
    else:
        remark = ("The student is at an early stage of developing professional communication skills. "
                  "It is strongly recommended to engage with role-play exercises, group discussions, "
                  "and formal writing workshops. Communication is a learnable skill — dedicated effort "
                  "over the coming weeks can bring about a remarkable improvement.")
        quote  = '"It usually takes more than three weeks to prepare a good impromptu speech." — Mark Twain'

    st.markdown(f"""
    <div class="remark-box">
      <strong style="color:#34d399;">Faculty Remarks:</strong><br><br>{remark}
    </div>
    <div style="background:rgba(99,102,241,0.08);border-left:3px solid #6366f1;border-radius:0 10px 10px 0;
                padding:12px 16px;margin:14px 0;font-size:0.85rem;color:#a5b4fc;font-style:italic;">
      {quote}
    </div>
    <div class="prof-footer">
      <strong style="color:#475569;">Evaluated by:</strong> Department of Communication Skills &amp; Professional Development<br>
      Assessment Standard: Industry-Aligned Placement Readiness Rubric · Academic Year 2024–25
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Save / Home / Retake ───────────────────────────────────────────
    col_use, col_home, col_retake = st.columns(3)
    with col_use:
        st.markdown('<div class="btn-use">', unsafe_allow_html=True)
        if st.button(f"✅  Send Score ({score_10}/10) → Placement Predictor", use_container_width=True):
            st.session_state["comm_score"] = score_10
            st.success(f"✅ Score {score_10}/10 saved! Go to **🏠 app** in the sidebar — Communication Skills auto-filled to {score_10}.")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_home:
        st.markdown('<div class="btn-home">', unsafe_allow_html=True)
        if st.button("🏠  Go to Home", use_container_width=True):
            st.switch_page("app.py")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_retake:
        if st.button("🔄  Retake Test", use_container_width=True):
            st.session_state.ct_answers   = {}
            st.session_state.ct_submitted = False
            st.session_state.ct_score     = None
            st.rerun()

    if st.session_state.get("comm_score") is not None:
        st.info(f"💾 Communication score **{st.session_state['comm_score']}/10** is saved. Go back to the main app to predict placement!")

    # ── Detailed answer review ─────────────────────────────────────────
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#34d399;margin-bottom:16px;">📋 Detailed Answer Review</p>', unsafe_allow_html=True)

    for i, q in enumerate(questions):
        user_idx   = st.session_state.ct_answers.get(i)
        is_correct = (user_idx == q["answer"])
        icon       = "✅" if is_correct else "❌"
        user_text  = q["options"][user_idx] if user_idx is not None else "Not answered"
        corr_text  = q["options"][q["answer"]]
        wrong_html = "" if is_correct else f" &nbsp;|&nbsp; Correct: <strong>{corr_text}</strong>"

        st.markdown(f"""
        <div class="q-card" style="border-color:{'rgba(52,211,153,0.25)' if is_correct else 'rgba(248,113,113,0.25)'};">
          <div class="q-num">Q{i+1} of 10</div>
          <div class="q-text">{q['text']}</div>
          <div style="background:{'rgba(52,211,153,0.08)' if is_correct else 'rgba(248,113,113,0.08)'};
                      border:1px solid {'rgba(52,211,153,0.3)' if is_correct else 'rgba(248,113,113,0.3)'};
                      border-radius:10px;padding:10px 14px;font-size:0.83rem;
                      color:{'#34d399' if is_correct else '#f87171'};">
            {icon} Your answer: <strong>{user_text}</strong>{wrong_html}
          </div>
          <div style="background:#13131e;border-left:3px solid #34d399;border-radius:0 10px 10px 0;
                      padding:10px 14px;margin-top:8px;font-size:0.82rem;color:#94a3b8;">
            💡 The best response is: <strong style="color:#34d399;">{corr_text}</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════
# QUESTION PAGE
# ══════════════════════════════════════════════════════════════════════
answered = len(st.session_state.ct_answers)
st.markdown(f'<p style="font-size:0.82rem;color:#64748b;margin-bottom:8px;">📝 {answered}/10 answered</p>', unsafe_allow_html=True)
st.progress(answered / 10)
st.markdown("<br>", unsafe_allow_html=True)

for i, q in enumerate(questions):
    st.markdown(f"""
    <div class="q-card">
      <div class="q-num">Question {i+1} of 10</div>
      <div class="q-text">{q['text']}</div>
    </div>
    """, unsafe_allow_html=True)

    prev   = st.session_state.ct_answers.get(i)
    choice = st.radio(
        f"Q{i+1}",
        options=list(range(len(q["options"]))),
        format_func=lambda x, opts=q["options"]: opts[x],
        index=prev if prev is not None else None,
        key=f"comm_q_{i}",
        label_visibility="collapsed"
    )
    if choice is not None:
        st.session_state.ct_answers[i] = choice
    st.markdown("<br>", unsafe_allow_html=True)

# ── Submit ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
answered_final = len(st.session_state.ct_answers)

if answered_final < 10:
    st.warning(f"⚠️ Please answer all 10 questions. ({10 - answered_final} remaining)")

st.markdown('<div class="btn-submit">', unsafe_allow_html=True)
submit = st.button("📋  Submit & Get Evaluation", use_container_width=True, type="primary",
                   disabled=(answered_final < 10))
st.markdown('</div>', unsafe_allow_html=True)

if submit:
    correct = sum(1 for i, q in enumerate(questions) if st.session_state.ct_answers.get(i) == q["answer"])
    st.session_state.ct_score     = correct
    st.session_state.ct_submitted = True
    st.rerun()

st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;padding:24px 0 8px;">
  ML Mini Project · Communication Skills Module · Evaluated against Industry Placement Standards
</div>
""", unsafe_allow_html=True)
