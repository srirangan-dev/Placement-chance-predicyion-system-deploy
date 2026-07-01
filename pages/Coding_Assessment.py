import streamlit as st

st.set_page_config(page_title="Coding Assessment", page_icon="💻", layout="wide")
st.markdown("""
<style>


@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #e8e6f0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1100px; }
.hero {
    background: linear-gradient(135deg, #0d0f1a 0%, #101a2e 60%, #0a1020 100%);
    border: 1px solid rgba(79,195,247,0.25); border-radius: 20px;
    padding: 36px 44px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute; top: -50px; right: -50px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(79,195,247,0.12) 0%, transparent 70%); border-radius: 50%;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(90deg, #4fc3f7, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0; line-height: 1.15;
}


.hero-sub { font-size: 0.95rem; color: #94a3b8; font-weight: 300; margin: 0 0 18px 0; }
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.badge { border-radius: 20px; padding: 4px 14px; font-size: 0.75rem; font-weight: 500; }
.badge.blue   { border: 1px solid rgba(79,195,247,0.4);  color: #4fc3f7; background: rgba(79,195,247,0.08); }
.badge.purple { border: 1px solid rgba(167,139,250,0.4); color: #a78bfa; background: rgba(167,139,250,0.08); }
.badge.green  { border: 1px solid rgba(52,211,153,0.4);  color: #34d399; background: rgba(52,211,153,0.08); }
.q-card { background: #111118; border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 24px 28px; margin-bottom: 20px; }
.q-card:hover { border-color: rgba(79,195,247,0.25); }
.q-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.q-num { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #4fc3f7; }
.q-tag { border-radius: 12px; padding: 3px 10px; font-size: 0.7rem; font-weight: 600; }
.tag-Easy   { background: rgba(52,211,153,0.12); color: #34d399; }
.tag-Medium { background: rgba(251,191,36,0.12);  color: #fbbf24; }
.tag-Hard   { background: rgba(248,113,113,0.12); color: #f87171; }
.q-text { font-size: 0.95rem; color: #e2e8f0; margin-bottom: 14px; line-height: 1.6; }
.code-block {
    background: #0d1117; border: 1px solid rgba(255,255,255,0.07); border-radius: 10px;
    padding: 14px 18px; margin-bottom: 14px; font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; color: #a9b7d0; line-height: 1.7; white-space: pre;
}
.score-card {
    background: linear-gradient(135deg, #0a1a2a, #0d1f35);
    border: 1px solid rgba(79,195,247,0.35); border-radius: 18px;
    padding: 36px; text-align: center; margin-bottom: 24px;
}
.score-big {
    font-family: 'Syne', sans-serif; font-size: 4rem; font-weight: 800;
    background: linear-gradient(90deg, #4fc3f7, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;
}
.score-label { font-size: 0.9rem; color: #64748b; margin-top: 8px; text-transform: uppercase; letter-spacing: 0.08em; }
.grade-badge { display: inline-block; margin-top: 14px; padding: 6px 22px; border-radius: 20px; font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; }
.metric-tile { background: #15151f; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 18px; text-align: center; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 1.8rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.72rem; color: #64748b; margin: 4px 0 0 0; text-transform: uppercase; letter-spacing: 0.06em; }
.ans-correct { background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.3); border-radius: 10px; padding: 10px 14px; margin-top: 10px; font-size: 0.83rem; color: #34d399; }
.ans-wrong   { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3); border-radius: 10px; padding: 10px 14px; margin-top: 10px; font-size: 0.83rem; color: #f87171; }
.ans-explain { background: #13131e; border-left: 3px solid #4fc3f7; border-radius: 0 10px 10px 0; padding: 10px 14px; margin-top: 8px; font-size: 0.82rem; color: #94a3b8; }
.stButton > button {
    background: linear-gradient(135deg, #0e5f87, #1a3a6e) !important;
    color: #4fc3f7 !important; border: 1px solid rgba(79,195,247,0.3) !important;
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
    {
        "id": 1, "difficulty": "Easy", "topic": "Arrays",
        "text": "What is the time complexity of accessing an element in an array by index?",
        "code": None,
        "options": ["O(n)", "O(log n)", "O(1)", "O(n²)"],
        "answer": 2,
        "explanation": "Array access by index is O(1) — direct memory address calculation: base_address + index × element_size."
    },
    {
        "id": 2, "difficulty": "Easy", "topic": "Python",
        "text": "What does the following Python code output?",
        "code": "x = [1, 2, 3, 4, 5]\nprint(x[1:4])",
        "options": ["[1, 2, 3, 4]", "[2, 3, 4]", "[1, 2, 3]", "[2, 3, 4, 5]"],
        "answer": 1,
        "explanation": "Python slicing x[1:4] returns elements at indices 1, 2, 3 (start inclusive, end exclusive) → [2, 3, 4]."
    },
    {
        "id": 3, "difficulty": "Medium", "topic": "Sorting",
        "text": "Which sorting algorithm has the best average-case time complexity?",
        "code": None,
        "options": ["Bubble Sort — O(n²)", "Merge Sort — O(n log n)", "Insertion Sort — O(n²)", "Selection Sort — O(n²)"],
        "answer": 1,
        "explanation": "Merge Sort guarantees O(n log n) in all cases. Quick Sort is O(n log n) average but O(n²) worst case."
    },
    {
        "id": 4, "difficulty": "Easy", "topic": "Python",
        "text": "What is the output of this Python snippet?",
        "code": "def foo(a, b=[]):\n    b.append(a)\n    return b\n\nprint(foo(1))\nprint(foo(2))",
        "options": ["[1]  [2]", "[1]  [1, 2]", "Error", "[1]  [2, 1]"],
        "answer": 1,
        "explanation": "Mutable default arguments are shared across calls. The list b is created once, so foo(2) appends to [1], giving [1, 2]."
    },
    {
        "id": 5, "difficulty": "Medium", "topic": "Data Structures",
        "text": "Which data structure uses LIFO (Last In, First Out) order?",
        "code": None,
        "options": ["Queue", "Stack", "Linked List", "Hash Map"],
        "answer": 1,
        "explanation": "Stack follows LIFO — the last element pushed is the first one popped."
    },
    {
        "id": 6, "difficulty": "Hard", "topic": "Recursion",
        "text": "What is the output of this recursive function?",
        "code": "def f(n):\n    if n <= 1:\n        return n\n    return f(n-1) + f(n-2)\n\nprint(f(6))",
        "options": ["8", "13", "5", "21"],
        "answer": 0,
        "explanation": "This is Fibonacci. f(6) = f(5)+f(4) = 5+3 = 8. Sequence: f(0)=0,f(1)=1,f(2)=1,f(3)=2,f(4)=3,f(5)=5,f(6)=8."
    },
    {
        "id": 7, "difficulty": "Medium", "topic": "OOP",
        "text": "Which OOP concept allows a class to inherit properties from multiple parent classes?",
        "code": None,
        "options": ["Encapsulation", "Polymorphism", "Multiple Inheritance", "Abstraction"],
        "answer": 2,
        "explanation": "Multiple Inheritance allows a class to inherit from more than one parent class."
    },
    {
        "id": 8, "difficulty": "Medium", "topic": "Complexity",
        "text": "What is the space complexity of a recursive Fibonacci function (without memoization)?",
        "code": None,
        "options": ["O(1)", "O(n)", "O(log n)", "O(2ⁿ)"],
        "answer": 1,
        "explanation": "The call stack depth reaches O(n) in the worst case, so space complexity is O(n)."
    },
    {
        "id": 9, "difficulty": "Hard", "topic": "Python",
        "text": "What does this code print?",
        "code": "a = [1, 2, 3]\nb = a\nb.append(4)\nprint(a)",
        "options": ["[1, 2, 3]", "[1, 2, 3, 4]", "Error", "[4, 1, 2, 3]"],
        "answer": 1,
        "explanation": "In Python, b = a does not create a copy — both point to the same list. Use b = a.copy() for a shallow copy."
    },
    {
        "id": 10, "difficulty": "Hard", "topic": "Algorithms",
        "text": "In a Binary Search Tree (BST), what is the average time complexity for search, insert, and delete?",
        "code": None,
        "options": ["O(n)", "O(1)", "O(log n)", "O(n log n)"],
        "answer": 2,
        "explanation": "In a balanced BST, each operation eliminates half the remaining nodes, giving O(log n) average."
    }
]

TOTAL_Q = len(QUESTIONS)

# ── Session State ──────────────────────────────────────────────────────────
if "ca_answers"   not in st.session_state: st.session_state.ca_answers   = {}
if "ca_submitted" not in st.session_state: st.session_state.ca_submitted = False
if "ca_score"     not in st.session_state: st.session_state.ca_score     = None

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <p class="hero-title">💻 Coding Skills Assessment</p>
  <p class="hero-sub">Answer 10 coding questions. Your score (0–10) will auto-fill the Coding Skills slider in the Placement Predictor.</p>
  <div class="hero-badges">
    <span class="badge blue">10 Questions</span>
    <span class="badge purple">1 Mark Each</span>
    <span class="badge green">Arrays · OOP · Algorithms · Python</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Results Page ───────────────────────────────────────────────────────────
if st.session_state.ca_submitted and st.session_state.ca_score is not None:
    score     = st.session_state.ca_score
    score_pct = score

    if score_pct >= 8:
        grade, gcol, gbg = "Excellent 🔥", "#34d399", "rgba(52,211,153,0.15)"
    elif score_pct >= 6:
        grade, gcol, gbg = "Good 👍",       "#4fc3f7", "rgba(79,195,247,0.15)"
    elif score_pct >= 4:
        grade, gcol, gbg = "Average 📈",    "#fbbf24", "rgba(251,191,36,0.15)"
    else:
        grade, gcol, gbg = "Needs Work 📚", "#f87171", "rgba(248,113,113,0.15)"

    st.markdown(f"""
    <div class="score-card">
      <div class="score-big">{score_pct} <span style="font-size:1.8rem;color:#334155;">/ 10</span></div>
      <div class="score-label">Coding Skills Score</div>
      <div class="grade-badge" style="background:{gbg};color:{gcol};border:1px solid {gcol}40;">{grade}</div>
    </div>
    """, unsafe_allow_html=True)

    correct_count = sum(1 for i, q in enumerate(QUESTIONS) if st.session_state.ca_answers.get(i) == q["answer"])
    wrong_count   = TOTAL_Q - correct_count

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#34d399">{correct_count}</p><p class="metric-label">Correct</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#f87171">{wrong_count}</p><p class="metric-label">Wrong</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#4fc3f7">{score_pct * 10}%</p><p class="metric-label">Accuracy</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-tile"><p class="metric-value" style="color:#a78bfa">{score_pct}/10</p><p class="metric-label">Your Score</p></div>', unsafe_allow_html=True)

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
        if st.button(f"✅  Send Score ({score_pct}/10) → Predictor", use_container_width=True):
            st.session_state["coding_score"] = score_pct
            st.success(f"✅ Score {score_pct}/10 saved! Go to **🏠 app** in the sidebar — Coding Skills auto-filled to {score_pct}.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_retake:
        if st.button("🔄  Retake Assessment", use_container_width=True):
            st.session_state.ca_answers   = {}
            st.session_state.ca_submitted = False
            st.session_state.ca_score     = None
            st.rerun()

    if st.session_state.get("coding_score") is not None:
        st.info(f"💾 Coding score **{st.session_state['coding_score']}/10** is saved. Go back to the main app to predict placement!")

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown('<p style="font-family:Syne,sans-serif;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;color:#4fc3f7;margin-bottom:16px;">📋 Detailed Review</p>', unsafe_allow_html=True)

    for i, q in enumerate(QUESTIONS):
        user_ans   = st.session_state.ca_answers.get(i)
        is_correct = (user_ans == q["answer"])
        icon       = "✅" if is_correct else "❌"
        diff_class = f"tag-{q['difficulty']}"
        code_html  = f'<div class="code-block">{q["code"]}</div>' if q["code"] else ""
        st.markdown(f"""
        <div class="q-card">
          <div class="q-header">
            <span class="q-num">Q{i+1} · {q['topic']}</span>
            <span class="q-tag {diff_class}">{q['difficulty']}</span>
          </div>
          <p class="q-text">{q['text']}</p>
          {code_html}
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
answered = len(st.session_state.ca_answers)
st.markdown(f'<p style="font-size:0.82rem;color:#64748b;margin-bottom:20px;">📝 {answered}/{TOTAL_Q} answered</p>', unsafe_allow_html=True)
st.progress(answered / TOTAL_Q)
st.markdown("<br>", unsafe_allow_html=True)

# ── Questions ──────────────────────────────────────────────────────────────
for i, q in enumerate(QUESTIONS):
    diff_class = f"tag-{q['difficulty']}"
    code_html  = f'<div class="code-block">{q["code"]}</div>' if q["code"] else ""

    st.markdown(f"""
    <div class="q-card">
      <div class="q-header">
        <span class="q-num">Question {i+1} of {TOTAL_Q} · {q['topic']}</span>
        <span class="q-tag {diff_class}">{q['difficulty']}</span>
      </div>
      <p class="q-text">{q['text']}</p>
      {code_html}
    </div>
    """, unsafe_allow_html=True)

    prev   = st.session_state.ca_answers.get(i)
    choice = st.radio(
        f"Select answer for Q{i+1}:",
        options=list(range(len(q["options"]))),
        format_func=lambda x, opts=q["options"]: opts[x],
        index=prev if prev is not None else None,
        key=f"ca_q_{i}",
        label_visibility="collapsed"
    )
    if choice is not None:
        st.session_state.ca_answers[i] = choice
    st.markdown("<br>", unsafe_allow_html=True)

# ── Submit ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
answered_final = len(st.session_state.ca_answers)

if answered_final < TOTAL_Q:
    st.warning(f"⚠️ Please answer all {TOTAL_Q} questions. ({TOTAL_Q - answered_final} remaining)")

st.markdown('<div class="btn-submit">', unsafe_allow_html=True)
submit = st.button("🚀  Submit Assessment", use_container_width=True, type="primary",
                   disabled=(answered_final < TOTAL_Q))
st.markdown('</div>', unsafe_allow_html=True)

if submit:
    correct = sum(1 for i, q in enumerate(QUESTIONS) if st.session_state.ca_answers.get(i) == q["answer"])
    st.session_state.ca_score     = correct
    st.session_state.ca_submitted = True
    st.rerun()

st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.78rem;padding:24px 0 8px;">
  Coding Assessment · 10 Questions · 10 Total Marks
</div>
""", unsafe_allow_html=True)
