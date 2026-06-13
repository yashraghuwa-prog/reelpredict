import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import cv2
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ReelPredict AI", page_icon="🎬", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #0f0f1a; color: #e2e8f0; }
.result-box {
    background: #1a1a2e; border: 1px solid #a78bfa;
    border-radius: 12px; padding: 1.5rem;
    text-align: center; margin-top: 1rem;
}
.metric-big { font-size: 2.8rem; font-weight: 700; color: #a78bfa; }
.range-label { font-size: 1rem; color: #94a3b8; margin-top: 0.4rem; }
.verdict-label { font-size: 1.2rem; margin-top: 0.6rem; }
.meta-box {
    background: #1a1a2e; border: 1px solid #2a2a3e;
    border-radius: 10px; padding: 0.9rem 1.2rem;
    margin-bottom: 1rem; font-size: 0.9rem; color: #94a3b8;
}
.meta-box span { color: #e2e8f0; font-weight: 600; }
.tip-box {
    background: #1a1a2e; border-left: 3px solid #a78bfa;
    border-radius: 0 8px 8px 0; padding: 0.6rem 1rem;
    margin: 0.4rem 0; font-size: 0.9rem; color: #cbd5e1;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def train_model():
    np.random.seed(42)
    N = 5000
    categories = ['dance','comedy','food','finance','fitness','fashion','travel','education','motivation','gaming']
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    category     = np.random.choice(categories, N)
    day_of_week  = np.random.choice(days, N)
    post_hour    = np.random.randint(0, 24, N)
    followers    = np.random.lognormal(mean=8.5, sigma=1.8, size=N).astype(int).clip(100, 50_000_000)
    reel_dur     = np.random.choice([7,15,30,45,60,90], N, p=[0.1,0.25,0.3,0.15,0.12,0.08])
    num_hashtags = np.random.randint(0, 31, N)
    caption_len  = np.random.randint(0, 300, N)
    has_music    = np.random.choice([0,1], N, p=[0.15,0.85])
    has_trend    = np.random.choice([0,1], N, p=[0.6,0.4])
    collab       = np.random.choice([0,1], N, p=[0.85,0.15])
    avg_past     = np.random.lognormal(mean=7, sigma=2, size=N).astype(int).clip(0, 10_000_000)
    engagement   = np.round(np.random.uniform(0.005, 0.15, N), 4)
    mentions     = np.random.randint(0, 10, N)
    sponsored    = np.random.choice([0,1], N, p=[0.9,0.1])

    def hour_score(h):
        if 6 <= h <= 9:   return 1.3
        if 12 <= h <= 14: return 1.2
        if 18 <= h <= 22: return 1.5
        return 0.8

    hour_scores = np.array([hour_score(h) for h in post_hour])
    day_scores  = {'Monday':0.9,'Tuesday':0.95,'Wednesday':1.0,'Thursday':1.1,'Friday':1.3,'Saturday':1.4,'Sunday':1.2}
    day_mult    = np.array([day_scores[d] for d in day_of_week])
    cat_mult    = {'dance':1.4,'comedy':1.35,'food':1.1,'finance':0.85,'fitness':1.15,
                   'fashion':1.2,'travel':1.05,'education':0.9,'motivation':1.0,'gaming':1.25}
    cat_scores  = np.array([cat_mult[c] for c in category])

    base = (followers * 0.08 + avg_past * 0.35 + engagement * followers * 2.5
            + has_trend * followers * 0.12 + has_music * 500
            + collab * followers * 0.05 + num_hashtags * 120
            + (reel_dur / 60) * 800 - abs(num_hashtags - 15) * 50
            + mentions * 200)

    views = (base * hour_scores * day_mult * cat_scores
             * np.random.lognormal(0, 0.5, N)).astype(int).clip(0, 100_000_000)

    df = pd.DataFrame({
        'category': category, 'day_of_week': day_of_week, 'post_hour': post_hour,
        'followers': followers, 'reel_duration_sec': reel_dur, 'num_hashtags': num_hashtags,
        'caption_length': caption_len, 'has_music': has_music, 'has_trending_audio': has_trend,
        'collab_post': collab, 'avg_past_views': avg_past, 'past_engagement_rate': engagement,
        'num_mentions': mentions, 'is_sponsored': sponsored, 'views': views
    })

    le_cat = LabelEncoder(); le_day = LabelEncoder()
    df['category_enc'] = le_cat.fit_transform(df['category'])
    df['day_enc']      = le_day.fit_transform(df['day_of_week'])
    df['log_views']    = np.log1p(df['views'])

    FEATURES = ['category_enc','day_enc','post_hour','followers','reel_duration_sec',
                'num_hashtags','caption_length','has_music','has_trending_audio',
                'collab_post','avg_past_views','past_engagement_rate','num_mentions','is_sponsored']

    X = df[FEATURES]; y = df['log_views']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    return {'model': model, 'le_cat': le_cat, 'le_day': le_day, 'features': FEATURES}


def get_video_duration(uploaded_file):
    """Write uploaded file to temp disk and read duration with OpenCV."""
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    uploaded_file.seek(0)  # reset so Streamlit can still show the video
    cap = cv2.VideoCapture(tmp_path)
    fps        = cap.get(cv2.CAP_PROP_FPS)
    frame_cnt  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    os.unlink(tmp_path)
    if fps > 0:
        return round(frame_cnt / fps)
    return None


def fmt(n):
    if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
    if n >= 1_000:     return f"{n/1_000:.1f}K"
    return str(n)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎬 ReelPredict AI")
st.caption("Upload your reel — get an instant view-count prediction powered by ML.")

with st.spinner("Warming up model (trains once on 5,000 records)..."):
    bundle = train_model()

model  = bundle['model']
le_cat = bundle['le_cat']
le_day = bundle['le_day']

st.divider()

# ── Step 1 — Upload ───────────────────────────────────────────────────────────
st.subheader("Step 1 · Upload your reel")
uploaded = st.file_uploader("Drag & drop your video here", type=["mp4","mov","avi","webm","mkv"])

duration_sec = None

if uploaded:
    # Show the video inline
    st.video(uploaded)

    # Auto-read duration
    with st.spinner("Reading video metadata..."):
        duration_sec = get_video_duration(uploaded)

    file_size_mb = round(uploaded.size / 1_000_000, 1)

    st.markdown(f"""
    <div class="meta-box">
        📁 <b>{uploaded.name}</b> &nbsp;|&nbsp;
        Size: <span>{file_size_mb} MB</span> &nbsp;|&nbsp;
        Duration: <span>{duration_sec if duration_sec else '?'}s</span> &nbsp;|&nbsp;
        Format: <span>{uploaded.type.split('/')[-1].upper()}</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Step 2 — Quick details ─────────────────────────────────────────────────
    st.subheader("Step 2 · A few quick details")

    col1, col2 = st.columns(2)
    with col1:
        category    = st.selectbox("Content category", le_cat.classes_)
        day         = st.selectbox("Day of posting", le_day.classes_,
                                   index=le_day.classes_.tolist().index("Friday"))
        hour        = st.slider("Post hour (24h)", 0, 23, 18)
        followers   = st.number_input("Your followers", min_value=100, max_value=50_000_000,
                                      value=10_000, step=500)
        avg_past    = st.number_input("Avg past views", min_value=0, max_value=10_000_000,
                                      value=5_000, step=500)

    with col2:
        hashtags    = st.slider("Hashtags", 0, 30, 12)
        caption_len = st.slider("Caption length (chars)", 0, 300, 100)
        engagement  = st.slider("Past engagement rate", 0.005, 0.15, 0.05, step=0.005)
        mentions    = st.slider("Mentions", 0, 10, 1)
        has_music   = st.toggle("Has background music", value=True)
        trending    = st.toggle("Uses trending audio", value=False)
        collab      = st.toggle("Collab post", value=False)
        sponsored   = st.toggle("Sponsored post", value=False)

    # Use auto-detected duration if available, else pick closest bucket
    if duration_sec:
        buckets = [7, 15, 30, 45, 60, 90]
        dur_input = min(buckets, key=lambda x: abs(x - duration_sec))
    else:
        dur_input = 30

    st.divider()

    # ── Step 3 — Predict ──────────────────────────────────────────────────────
    if st.button("🚀 Predict views", use_container_width=True, type="primary"):
        features = [[
            le_cat.transform([category])[0],
            le_day.transform([day])[0],
            hour, followers, dur_input, hashtags, caption_len,
            int(has_music), int(trending), int(collab),
            avg_past, engagement, mentions, int(sponsored)
        ]]

        log_pred = model.predict(features)[0]
        pred = int(np.expm1(log_pred))
        low  = int(pred * 0.6)
        high = int(pred * 1.4)

        verdict = ("🔥 Viral Potential!" if pred > 100_000
                   else "📈 Good Reach"   if pred > 20_000
                   else "📊 Moderate Reach" if pred > 5_000
                   else "🌱 Limited Reach")
        color   = ("#34d399" if pred > 100_000 else "#a78bfa"
                   if pred > 20_000 else "#fbbf24" if pred > 5_000 else "#94a3b8")

        st.markdown(f"""
        <div class="result-box">
          <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:0.3rem">Predicted view count</div>
          <div class="metric-big" style="color:{color}">{fmt(pred)}</div>
          <div class="range-label">Estimated range: {fmt(low)} — {fmt(high)}</div>
          <div class="verdict-label" style="color:{color}">{verdict}</div>
          {'<div style="font-size:0.82rem;color:#64748b;margin-top:0.5rem">Duration auto-detected: ' + str(duration_sec) + 's → mapped to ' + str(dur_input) + 's bucket</div>' if duration_sec else ''}
        </div>
        """, unsafe_allow_html=True)

        # Tips
        tips = []
        if not trending:   tips.append("🎵 Use trending audio — it's the single biggest reach booster")
        if hour < 6 or hour > 22: tips.append("⏰ Post between 6 pm – 10 pm for peak engagement")
        if hashtags < 8:   tips.append("#️⃣ Add 10–15 targeted hashtags")
        if dur_input < 15: tips.append("⏱ Reels 15–30 s tend to get replayed more — try a slightly longer cut")
        if not collab:     tips.append("🤝 A collab post can double your reach overnight")

        if tips:
            st.markdown("#### 💡 Tips to improve your reach")
            for t in tips:
                st.markdown(f'<div class="tip-box">{t}</div>', unsafe_allow_html=True)

else:
    st.info("👆 Upload a reel above to get started.")
