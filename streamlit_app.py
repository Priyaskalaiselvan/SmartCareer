# app.py - Level 3 UI (Advanced, stable, no-HTML)
import streamlit as st
import pandas as pd
import numpy as np
import graphviz
import json
from io import BytesIO
from resume_parser import parse_resume


# Defensive imports for optional resume parsing
try:
    import PyPDF2
    _has_pypdf2 = True
except Exception:
    _has_pypdf2 = False

# Embedding & index libs
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    st.error("Missing `sentence-transformers`. Install with `pip install sentence-transformers`")
    raise

try:
    import faiss
except Exception:
    st.error("Missing `faiss-cpu`. Install with `pip install faiss-cpu`")
    raise

# -------------------- CACHING & RESOURCE SETUP -------------------- #
@st.cache_resource
def load_courses(path="courses.csv"):
    df_local = pd.read_csv(path)
    for col in ['title','provider','duration','prerequisites','skill_tags','level','link']:
        if col not in df_local.columns:
            df_local[col] = ""
    df_local['prerequisites'] = df_local['prerequisites'].fillna("").apply(lambda x: x.strip() if isinstance(x, str) else "")
    df_local['prerequisites'] = df_local['prerequisites'].replace("", "No prerequisites")
    df_local['skill_tags'] = df_local['skill_tags'].fillna("").apply(lambda x: x.strip() if isinstance(x, str) else "general")
    df_local['combined_text'] = (
        df_local['title'].astype(str) + " | " +
        df_local['provider'].astype(str) + " | " +
        df_local['level'].astype(str) + " | " +
        df_local['skill_tags'].astype(str) + " | " +
        df_local['prerequisites'].astype(str)
    )
    return df_local

@st.cache_resource
def load_model():
    # Force CPU device to avoid meta tensor issues
    return SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

# Note: use leading underscore to avoid hashing model object
@st.cache_data
def compute_embeddings(texts, _model):
    emb = _model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)
    return emb.astype('float32')

@st.cache_resource
def build_faiss_index(_emb_matrix):
    dim = _emb_matrix.shape[1]
    index_local = faiss.IndexFlatL2(dim)
    index_local.add(_emb_matrix)
    return index_local

# -------------------- LOAD ONCE -------------------- #
with st.spinner("Loading course catalog and model..."):
    df = load_courses("courses.csv")
    model = load_model()
    emb_matrix = compute_embeddings(df['combined_text'], model)
    index = build_faiss_index(emb_matrix)

# Precompute available skill tags for chips/filtering
_all_tags = sorted({tag.strip().lower() for cell in df['skill_tags'].astype(str) for tag in cell.split(",")})

# -------------------- UTIL FUNCTIONS -------------------- #
def extract_text_from_pdf(file_bytes):
    if not _has_pypdf2:
        st.warning("PyPDF2 not installed — resume PDF parsing not available.")
        return ""
    try:
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        pages = []
        for pg in reader.pages:
            pages.append(pg.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        st.warning(f"PDF parsing failed: {e}")
        return ""

def simple_skill_extractor(text, candidate_skills=None, top_n=20):
    # Very simple extractor: count token occurrences and match against candidate skills (if provided)
    text_lower = text.lower()
    tokens = [t.strip() for t in text_lower.replace("\n"," ").split() if len(t)>1]
    freq = {}
    for tok in tokens:
        freq[tok] = freq.get(tok,0) + 1
    # If candidate skills provided (e.g., tags), match them
    found = []
    if candidate_skills:
        for skill in candidate_skills:
            if skill.lower() in text_lower:
                found.append(skill.lower())
    # fallback: return top tokens (best-effort)
    if not found:
        top = sorted(freq.items(), key=lambda x: -x[1])[:top_n]
        found = [t for t,_ in top if len(t)>2]
    # dedupe and return reasonable set
    seen = []
    for s in found:
        if s not in seen:
            seen.append(s)
    return seen[:20]

def generate_rationale(user_skills, course_skill_tags, prerequisites):
    user_skills = [s.lower().strip() for s in user_skills]
    course_skills = [s.lower().strip() for s in str(course_skill_tags).split(",")]
    matched = [c for c in course_skills if any(c in us or us in c for us in user_skills)]
    if matched:
        r1 = f"Matches your skills: {', '.join(matched)}."
    else:
        r1 = "This course adds relevant skills you don't yet list."
    if prerequisites.lower() != "no prerequisites":
        r2 = f"Fills prerequisite: {prerequisites}."
    else:
        r2 = "No prerequisites — good quick start."
    return r1 + " " + r2

def assign_timeline(duration_text, level):
    text = str(duration_text).lower()
    weeks = 6
    try:
        if "month" in text:
            weeks = int(text.split()[0]) * 4
        elif "hour" in text:
            weeks = max(1, int(text.split()[0]) // 6)
        else:
            import re
            m = re.search(r'(\d+)', text)
            if m:
                weeks = int(m.group(1))
    except Exception:
        weeks = 6
    if weeks <= 8 and level.lower() == "beginner":
        return "Short-term (1–2 months)"
    if level.lower() == "advanced" or weeks > 12:
        return "Long-term (3–6 months)"
    return "Medium-term (2–3 months)"

def generate_learning_path(selected_df):
    path = {"Foundation": [], "Skill Builder": [], "Advanced": []}
    for _, r in selected_df.iterrows():
        lvl = str(r['level']).lower()
        prereq = str(r['prerequisites']).lower()
        if lvl == "beginner" and "no prerequisites" in prereq:
            path["Foundation"].append(r['title'])
        elif lvl == "intermediate":
            path["Skill Builder"].append(r['title'])
        else:
            path["Advanced"].append(r['title'])
    return path

def recommend_courses(user_profile_text, user_skills_list, top_k=6):
    user_emb = model.encode([user_profile_text], convert_to_numpy=True)[0].astype('float32')
    D, I = index.search(np.array([user_emb]), top_k)
    selected = df.iloc[I[0]].copy().reset_index(drop=True)
    selected['score'] = (1 / (1 + D[0])) * 100
    selected['rationale'] = selected.apply(
        lambda r: generate_rationale(user_skills_list, r['skill_tags'], r['prerequisites']), axis=1)
    selected['timeline'] = selected.apply(lambda r: assign_timeline(r['duration'], r['level']), axis=1)
    return selected, generate_learning_path(selected)

# -------------------- USER INTERFACE (LEVEL 3) -------------------- #
st.set_page_config(page_title="AI Learning Path", layout="wide")

# Top bar + mode toggle
st.title("🎯 AI Course Recommender — Level 3")
col_top = st.columns([0.7, 0.3])
with col_top[1]:
    visual_mode = st.selectbox("Visual mode", ["Light", "Dark (compact)"])

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    top_k = st.slider("Recommendations (top K)", 3, 12, 6)
    tag_filter = st.multiselect("Filter by skill tags (chips)", options=_all_tags, default=None)
    domain_quick = st.selectbox("Quick domain", ["All","Data","AI","Cloud","Web","DevOps","Security","Design","Business"])
    st.markdown("---")
    st.info("You can upload a resume (PDF or TXT) to auto-fill skills.")

# Tabs: Courses / Path / Export
tabs = st.tabs(["🧑‍💼 Profile", "📘 Courses", "🧭 Path & Flowchart", "⬇ Export"])

# -------------------- Profile Tab -------------------- #


   # -------------------- PROFILE TAB -------------------- #
from resume_parser import parse_resume   # <-- uses lightweight parser

with tabs[0]:
    st.header("🧑‍💼 Tell us about yourself")

    # -------- FORM -------- #
    with st.form("profile_form_level3"):
        col1, col2 = st.columns(2)

        # --- Column 1 (Education / Experience) ---
        with col1:
            education = st.selectbox(
                "Education Level",
                ["High School", "Diploma", "Bachelor's", "Master's", "PhD"]
            )
            degree = st.text_input("Degree / Major", placeholder="B.Tech CSE")
            experience = st.selectbox(
                "Experience",
                ["No experience", "0-1 years", "1-3 years", "3-5 years", "5+ years"]
            )
            availability = st.selectbox(
                "Weekly Study Time",
                ["<5 hrs", "5-10 hrs", "10-20 hrs", "20+ hrs"]
            )

        # --- Column 2 (Skills + Career Goals) ---
        with col2:
            tech_input = st.text_input(
                "Technical Skills (comma separated)",
                placeholder="Python, SQL, Java"
            )
            soft_input = st.text_input(
                "Soft Skills (comma separated)",
                placeholder="Communication, teamwork"
            )
            career_goal = st.text_input("Career Goal", placeholder="Data Analyst / ML Engineer")
            preferred_duration = st.selectbox(
                "Preferred Duration",
                ["1 month", "2 months", "3 months", "4–6 months", "6–12 months"]
            )

        submitted = st.form_submit_button("Save Profile")

    # =========================================================
    #                📄 LIGHTWEIGHT RESUME UPLOAD
    # =========================================================
    st.markdown("### 📄 Resume Upload (Optional)")
    uploaded = st.file_uploader("Upload your resume (.pdf only)", type=["pdf"])

    parsed_resume = None

    if uploaded:
        with st.spinner("Extracting details from resume..."):
            file_bytes = uploaded.read()

            # ---- Parse resume using the lightweight parser ----
            parsed_resume = parse_resume(file_bytes)

            if parsed_resume and parsed_resume.get("raw_text"):
                st.success("Resume parsed successfully!")

                # ----- SHOW PARSED DETAILS -----
                st.write("#### 🧠 Extracted Skills")
                st.write(", ".join(parsed_resume.get("skills", [])))

                st.write("#### 🛠 Technical Skills Detected")
                st.write(", ".join(parsed_resume.get("tech_skills", [])))

                st.write("#### 🤝 Soft Skills Detected")
                st.write(", ".join(parsed_resume.get("soft_skills", [])))

                st.write("#### 🎓 Education Found")
                st.write(", ".join(parsed_resume.get("education", [])))

                st.write("#### 📜 Certifications Found")
                st.write(parsed_resume.get("certifications", []))

                st.write("#### 💼 Experience Extracted")
                st.write(parsed_resume.get("experience_years", "Not specified"))

                # ---- AUTO-FILL USER TECH SKILLS IF EMPTY ----
                auto_tech = parsed_resume.get("tech_skills", [])
                if tech_input.strip() == "" and auto_tech:
                    tech_input = ", ".join(auto_tech[:10])
                    st.info("Auto-filled technical skills based on resume.")


    # =========================================================
    #        FINAL SKILL MERGING (USER INPUT + RESUME)
    # =========================================================
    final_skills = set()

    # Add user-entered skills
    final_skills.update([s.strip().lower() for s in tech_input.split(",") if s.strip()])

    # Add resume-detected skills
    if parsed_resume:
        final_skills.update([s.strip().lower() for s in parsed_resume.get("skills", [])])

    # Final clean list used for recommendations
    user_skills = list(final_skills)

    # Build unified profile text for embeddings
    user_profile_text = f"""
    Education: {education}. Degree: {degree}. Experience: {experience}.
    Technical skills: {tech_input}. Soft skills: {soft_input}.
    Career Goal: {career_goal}. Preferred Duration: {preferred_duration}.
    Weekly Availability: {availability}.
    """

    if submitted:
        st.success("Profile saved successfully! Now open the **Courses** tab to continue.")

# -------------------- Courses Tab -------------------- #
# -------------------- COURSES TAB -------------------- #
with tabs[1]:
    st.header("📘 Recommended Courses")

    # If profile isn't submitted yet
    if 'submitted' not in locals() or not submitted:
        st.info("Fill in your profile and click **Save Profile**, then return here.")
    else:
        # Get recommendations
        rec_df, learning_path = recommend_courses(user_profile_text, user_skills, top_k=top_k)

        # Apply sidebar filters
        if tag_filter:
            rec_df = rec_df[
                rec_df['skill_tags'].str.lower()
                .apply(lambda cell: any(tag in cell for tag in tag_filter))
            ]

        if domain_quick != "All":
            rec_df = rec_df[rec_df['skill_tags'].str.contains(domain_quick, case=False)]

        if rec_df.empty:
            st.warning("No results match your filters. Try removing some filters.")
        else:
            # Display courses one by one
            for i, row in rec_df.iterrows():

                st.subheader(f"{row['title']}")
                meta = f"{row['provider']} • {row['level']} • {row['duration']}"
                st.caption(meta)

                # --- Tags ---
                tags_clean = [t.strip() for t in row["skill_tags"].split(",") if t.strip()]
                tags_line = ", ".join([f"🔹 {t}" for t in tags_clean])
                st.write(f"**Skills Covered:** {tags_line}")

                # --- Other details ---
                st.write(f"**Prerequisites:** {row['prerequisites']}")
                st.write(f"**Timeline:** {row['timeline']}")

                # --- Score with progress bar ---
                st.write("**Fit Score:**")
                st.progress(int(row['score']))

                # Color-coded match strength
                if row['score'] >= 80:
                    st.success("High fit")
                elif row['score'] >= 55:
                    st.warning("Medium fit")
                else:
                    st.info("Low fit")

                # --- Rationale ---
                st.info(f"**Why this course?**\n{row['rationale']}")

                # --- NEW: Clean official button ---
                st.link_button("🔗 Open Course", row["link"])

                st.write("---")


# -------------------- Path & Flowchart Tab -------------------- #
with tabs[2]:
    st.header("Learning Path & Flowchart")
    try:
        # Ensure rec_df exists (recommendation was run)
        rec_df  # may raise if not present
    except Exception:
        st.info("Generate recommendations first from the Profile tab.")
    else:
        path = generate_learning_path(rec_df)
        st.subheader("Structured Path")
        for stage, items in path.items():
            if items:
                if stage == "Foundation":
                    st.markdown(f"### 🔰 {stage}")
                elif stage == "Skill Builder":
                    st.markdown(f"### 🔧 {stage}")
                else:
                    st.markdown(f"### 🏁 {stage}")
                for c in items:
                    st.write(f"- {c}")
                st.write("")
        # Flowchart
        st.subheader("Visual Flowchart")
        dot = graphviz.Digraph()
        dot.attr(rankdir="LR")
        # Add nodes
        for stg in ["Foundation","Skill Builder","Advanced"]:
            for c in path.get(stg, []):
                dot.node(c, c, shape="box")
        # chain nodes
        last = None
        for stg in ["Foundation","Skill Builder","Advanced"]:
            for c in path.get(stg, []):
                if last:
                    dot.edge(last, c)
                last = c
        st.graphviz_chart(dot)

# -------------------- Export Tab -------------------- #
with tabs[3]:
    st.header("Export & Save")
    if 'rec_df' not in locals():
        st.info("No recommendations yet. Generate recommendations first.")
    else:
        export_payload = {
            "profile": {
                "education": education, "degree": degree, "experience": experience,
                "skills": user_skills, "goal": career_goal, "preferred_duration": preferred_duration
            },
            "recommendations": rec_df.to_dict(orient="records"),
            "learning_path": generate_learning_path(rec_df)
        }
        b = BytesIO()
        b.write(json.dumps(export_payload, indent=2).encode())
        b.seek(0)
        st.download_button("Download JSON", data=b, file_name="learning_path.json", mime="application/json")
        st.markdown("---")
        st.write("Or copy JSON preview:")
        st.code(json.dumps(export_payload, indent=2)[:4000] + ("\n... (truncated)" if len(json.dumps(export_payload))>4000 else ""), language='json')

# -------------------- FOOTER -------------------- #
st.sidebar.markdown("---")
st.sidebar.write("Level 3 UI • Stable • No raw HTML • Cached model & FAISS")
