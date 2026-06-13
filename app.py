import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="ReelPredict AI", page_icon="🎬", layout="centered")

st.markdown("""
<style>
body { background-color: #0f0f1a; }
.main { background-color: #0f0f1a; }
.stApp { background-color: #0f0f1a; color: #e2e8f0; }
.result-box { background: #1a1a2e; border: 1px solid #a78bfa; border-radius: 12px; padding: 1.5rem; text-align: center; margin-top: 1rem; }
.metric-big { font-size: 2.5rem; font-weight: 700; color: #a78bfa; }
.range-label { font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

with open('/home/claude/project/model.pkl', 'rb') as f:
    bundle = pickle.load(f)
model = bundle['model']
le_cat = bundle['le_cat']
le_day = bundle['le_day']

st.title("🎬 ReelPredict AI")
st.caption("Predict how many views your Instagram Reel will get — before you post it.")
st.divider()

col1, col2 = st.columns(2)
with col1:
    category = st.selectbox("Content Category", le_cat.classes_)
    day = st.selectbox("Day of Posting", le_day.classes_)
    hour = st.slider("Post Hour (24h)", 0, 23, 18)
    followers = st.number_input("Your Followers", min_value=100, max_value=50_000_000, value=10000, step=500)
    duration = st.selectbox("Reel Duration (sec)", [7, 15, 30, 45, 60, 90], index=2)
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
      <div class="range-label">Estimated range: {fmt(low)} – {fmt(high)}</div>
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
