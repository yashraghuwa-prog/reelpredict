import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 5000

categories = ['dance', 'comedy', 'food', 'finance', 'fitness', 'fashion', 'travel', 'education', 'motivation', 'gaming']
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Base features
category = np.random.choice(categories, N)
day_of_week = np.random.choice(days, N)
post_hour = np.random.randint(0, 24, N)
followers = np.random.lognormal(mean=8.5, sigma=1.8, size=N).astype(int).clip(100, 50_000_000)
reel_duration = np.random.choice([7,15,30,45,60,90], N, p=[0.1,0.25,0.3,0.15,0.12,0.08])
num_hashtags = np.random.randint(0, 31, N)
caption_length = np.random.randint(0, 300, N)
has_music = np.random.choice([0, 1], N, p=[0.15, 0.85])
has_trending_audio = np.random.choice([0, 1], N, p=[0.6, 0.4])
collab_post = np.random.choice([0, 1], N, p=[0.85, 0.15])
avg_past_views = np.random.lognormal(mean=7, sigma=2, size=N).astype(int).clip(0, 10_000_000)
past_engagement_rate = np.round(np.random.uniform(0.005, 0.15, N), 4)
num_mentions = np.random.randint(0, 10, N)
is_sponsored = np.random.choice([0, 1], N, p=[0.9, 0.1])

# Peak hour bonus (6-9am, 12-2pm, 6-10pm)
def hour_score(h):
    if 6 <= h <= 9: return 1.3
    if 12 <= h <= 14: return 1.2
    if 18 <= h <= 22: return 1.5
    return 0.8

hour_scores = np.array([hour_score(h) for h in post_hour])

# Day bonus
day_scores = {'Monday':0.9,'Tuesday':0.95,'Wednesday':1.0,'Thursday':1.1,'Friday':1.3,'Saturday':1.4,'Sunday':1.2}
day_mult = np.array([day_scores[d] for d in day_of_week])

# Category multiplier
cat_mult = {'dance':1.4,'comedy':1.35,'food':1.1,'finance':0.85,'fitness':1.15,
            'fashion':1.2,'travel':1.05,'education':0.9,'motivation':1.0,'gaming':1.25}
cat_scores = np.array([cat_mult[c] for c in category])

# Compute views with realistic noise
base = (
    followers * 0.08
    + avg_past_views * 0.35
    + past_engagement_rate * followers * 2.5
    + has_trending_audio * followers * 0.12
    + has_music * 500
    + collab_post * followers * 0.05
    + num_hashtags * 120
    + (reel_duration / 60) * 800
    - abs(num_hashtags - 15) * 50
    + num_mentions * 200
)

views = (base * hour_scores * day_mult * cat_scores * np.random.lognormal(0, 0.5, N)).astype(int).clip(0, 100_000_000)

df = pd.DataFrame({
    'category': category,
    'day_of_week': day_of_week,
    'post_hour': post_hour,
    'followers': followers,
    'reel_duration_sec': reel_duration,
    'num_hashtags': num_hashtags,
    'caption_length': caption_length,
    'has_music': has_music,
    'has_trending_audio': has_trending_audio,
    'collab_post': collab_post,
    'avg_past_views': avg_past_views,
    'past_engagement_rate': past_engagement_rate,
    'num_mentions': num_mentions,
    'is_sponsored': is_sponsored,
    'views': views
})

os.makedirs('/home/claude/project', exist_ok=True)
df.to_csv('/home/claude/project/instagram_reels_data.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.describe())
print("\nSample rows:")
print(df.head(3).to_string())
