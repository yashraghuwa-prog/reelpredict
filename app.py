import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ReelPredict AI", page_icon="🎬", layout="centered")

st.markdown("""
<style>
.stApp { background-color: #0f0f1a; color: #e2e8f0; }
.result-box { background: #1a1a2e; border: 1px solid #a78bfa; border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
.metric-big { font-size: 2.5rem; font-weight: 700; color: #a78bfa; }
.range-label { font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = 'model.pkl'

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)

    # Generate dataset on the fly
    np.random.seed(42)
    N = 5000
    categories = ['dance','comedy','food','finance','fitness','fashion','travel','education','motivation','gaming']
    days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    category = np.random.choice(categories, N)
    day_of_week = np.random.choice(days, N)
    post_hour = np.random.randint(0, 24, N)
    followers = np.random.lognormal(mean=8.5, sigma=1.8, size=N).astype(int).clip(100, 50_000_000)
    reel_duration = np.random.choice([7,15,30,45,60,90], N, p=[0.1,0.25,0.3,0.15,0.12,0.08])
    num_hashtags = np.random.randint(0, 31, N)
    caption_length = np.random.randint(0, 300, N)
    has_music = np.random.choice([0,1], N, p=[0.15,0.85])
    has_trending_audio = np.random.choice([0,1], N, p=[0.6,0.4])
    collab_post = np.random.choice([0,1], N, p=[0.85,0.15])
    avg_past_views = np.random.lognormal(mean=7, sigma=2, size=N).astype(int).clip(0, 10_000_000)
    past_engagement_rate = np.round(np.random.uniform(0.005, 0.15, N), 4)
    num_mentions = np.random.randint(0, 10, N)
    is_sponsored = np.random.choice([0,1], N, p=[0.9,0.1])

    def hour_score(h):
        if 6 <= h <= 9: return 1.3
        if 12 <= h <= 14: return 1.2
        if 18 <= h <= 22: return 1.5
        return 0.8

    hour_scores = np.array([hour_score(h) for h in post_hour])
    day_scores = {'Monday':0.9,'Tuesday':0.95,'Wednesday':1.0,'Thursday':1.1,'Friday':1.3,'Saturday':1.4,'Sunday':1.2}
    day_mult = np.array([day_scores[d] for d in day_of_week])
    cat_mult = {'dance':1.4,'comedy':1.35,'food':1.1,'finance':0.85,'fitness':1.15,'fashion':1.2,'travel':1.05,'education':0.9,'motivation':1.0,'gaming':1.25}
    cat_scores = np.array([cat_mult[c] for c in category])

    base = (followers * 0.08 + avg_past_views * 0.35 + past_engagement_rate * followers * 2.5
            + has_trending_audio * followers * 0.12 + has_music * 500
            + collab_post * followers * 0.05 + num_hashtags * 120
            + (reel_duration / 60) * 800 - abs(num_hashtags - 15) * 50
            + num_mentions * 200)

    views = (base * hour_scores * day_mult * cat_scores * np.random.lognormal(0, 0.5, N)).astype(int).clip(0, 100_000_000)

    df = pd.DataFrame({
        'category': category, 'day_of_week': day_of_week, 'post_hour': post_hour,
        'followers': followers, 'reel_duration_sec': reel_duration, 'num_hashtags': num_hashtags,
        'caption_length': caption_length, 'has_music': has_music, 'has_trending_audio': has_trending_audio,
        'collab_post': collab_post, 'avg_past_views': avg_past_views,
        'past_engagement_rate': past_engagement_rate, 'num_mentions': num_mentions,
        'is_sponsored': is_sponsored, 'views': views
    })

    le_cat = LabelEncoder()
    le_day = LabelEncoder()
    df['category_enc'] = le_cat.fit_transform(df['category'])
    df['day_enc'] = le_day.fit_transform(df['day_of_week'])
    df['log_views'] = np.log1p(df['views'])

    FEATURES = ['category_enc','day_enc','post_hour','followers','reel_duration_sec',
                'num_hashtags','caption_length','has_music','has_trending_audio',
                'collab_post','avg_past_views','past_engagement_rate','num_mentions','is_sponsored']

    X = df[FEATURES]
    y = df['log_views']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    bundle = {'model': model, 'le_cat': le_cat, 'le_day': le_day, 'features': FEATURES}
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f)

    return bundle

# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🎬 ReelPredict AI")
st.caption("Predict how many views your Instagram Reel will get — before you post it.")

with st.spinner("Loading model... (first run trains on 5,000 records, ~15 seconds)"):
    bundle = load_or_train_model()

model = bundle['model']
le_cat = bundle['le_cat']
le_day = bundle['le_day']

st.divider()
col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Content Category", le_cat.classes_)
    day = st.selectbox("Day of Posting", le_day.classes_)
    hour = st.slider("Post Hour (24h)", 0, 23, 18)
    followers = st.number_input("Your Followers", min_value=100, max_value=50_000_000, value=10000, step=500)
    duration = st.selectbox("Reel Duration (sec)", [7,15,30,45,60,90], index=2)
    avg_past = st.number_input("Avg Past Views", min_value=0, max_value=10_000_000, value=5000, step=500)

with col2:
    hashtags = st.slider("Number of Hashtags", 0, 30, 12)
    caption_len = st.slider("Caption Length (chars)", 0, 300, 100)
    engagement = st.slider("Past Engagement Rate", 0.005, 0.15, 0.05, step=0.005)
    mentions = st.slider("Number of Mentions", 0, 10, 1)
    has_music = st.toggle("Has Background Music", value=True)
    trending_audio = st.toggle("Uses Trending Audio", value=False)
    collab = st.toggle("Collaboration Post", value=False)
    sponsored = st.toggle("Sponsored Post", value=False)

st.divider()
if st.button("🚀 Predict Views", use_container_width=True, type="primary"):
    features = [[
        le_cat.transform([category])[0],
        le_day.transform([day])[0],
        hour, followers, duration, hashtags, caption_len,
        int(has_music), int(trending_audio), int(collab),
        avg_past, engagement, mentions, int(sponsored)
    ]]
    log_pred = model.predict(features)[0]
    pred = int(np.expm1(log_pred))
    low = int(pred * 0.6)
    high = int(pred * 1.4)

    def fmt(n):
        if n >= 1_000_000: return f"{n/1_000_000:.1f}M"
        if n >= 1_000: return f"{n/1_000:.1f}K"
        return str(n)

    verdict = "🔥 Viral Potential!" if pred > 100000 else "📈 Good Reach" if pred > 20000 else "📊 Moderate Reach" if pred > 5000 else "🌱 Limited Reach"
    color = "#34d399" if pred > 100000 else "#a78bfa" if pred > 20000 else "#fbbf24" if pred > 5000 else "#94a3b8"

    st.markdown(f"""
    <div class="result-box">
      <div style="font-size:1rem;color:#94a3b8;margin-bottom:0.5rem">Predicted View Count</div>
      <div class="metric-big" style="color:{color}">{fmt(pred)}</div>
      <div class="range-label">Estimated range: {fmt(low)} — {fmt(high)}</div>
      <div style="font-size:1.2rem;margin-top:0.75rem">{verdict}</div>
    </div>
    """, unsafe_allow_html=True)

    tips = []
    if not trending_audio: tips.append("🎵 Use trending audio to boost reach")
    if hour < 6 or hour > 22: tips.append("⏰ Post between 6pm–10pm for peak engagement")
    if hashtags < 8: tips.append("#️⃣ Add 10–15 relevant hashtags")
    if not collab: tips.append("🤝 Try a collab post for extra exposure")
    if tips:
        st.markdown("**💡 Tips to improve your reach:**")
        for t in tips: st.markdown(f"- {t}")
