import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('/home/claude/project/instagram_reels_data.csv')

# Encode categoricals
le_cat = LabelEncoder()
le_day = LabelEncoder()
df['category_enc'] = le_cat.fit_transform(df['category'])
df['day_enc'] = le_day.fit_transform(df['day_of_week'])

# Log transform target (views are skewed)
df['log_views'] = np.log1p(df['views'])

FEATURES = ['category_enc','day_enc','post_hour','followers','reel_duration_sec',
            'num_hashtags','caption_length','has_music','has_trending_audio',
            'collab_post','avg_past_views','past_engagement_rate','num_mentions','is_sponsored']

X = df[FEATURES]
y = df['log_views']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_actual = np.expm1(y_test)

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print(f"R² Score:  {r2:.4f}")
print(f"MAE:       {mae:,.0f} views")
print(f"RMSE:      {rmse:,.0f} views")

# Save model + encoders
with open('/home/claude/project/model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'le_cat': le_cat, 'le_day': le_day, 'features': FEATURES}, f)

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12), facecolor='#0f0f1a')
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

PURPLE = '#a78bfa'; TEAL = '#34d399'; CORAL = '#f87171'; GOLD = '#fbbf24'
DARK_BG = '#0f0f1a'; CARD_BG = '#1a1a2e'; GRID = '#2a2a3e'; TEXT = '#e2e8f0'; MUTED = '#94a3b8'

def style_ax(ax, title):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=TEXT, fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

# 1. Actual vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
sample = np.random.choice(len(y_actual), 300, replace=False)
ax1.scatter(y_actual.iloc[sample], y_pred[sample], alpha=0.5, color=PURPLE, s=18, edgecolors='none')
lims = [0, min(y_actual.max(), y_pred.max())]
ax1.plot(lims, lims, color=TEAL, linewidth=1.5, linestyle='--', label='Perfect fit')
style_ax(ax1, 'Actual vs Predicted Views')
ax1.set_xlabel('Actual Views', color=MUTED, fontsize=9)
ax1.set_ylabel('Predicted Views', color=MUTED, fontsize=9)
ax1.legend(fontsize=8, facecolor=CARD_BG, labelcolor=TEXT, edgecolor=GRID)
ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1000:.0f}K'))

# 2. Feature Importance
ax2 = fig.add_subplot(gs[0, 1])
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(10)
nice_names = {'category_enc':'Category','day_enc':'Day of Week','post_hour':'Post Hour',
              'followers':'Followers','reel_duration_sec':'Duration','num_hashtags':'Hashtags',
              'caption_length':'Caption Length','has_music':'Has Music','has_trending_audio':'Trending Audio',
              'collab_post':'Collab Post','avg_past_views':'Avg Past Views',
              'past_engagement_rate':'Engagement Rate','num_mentions':'Mentions','is_sponsored':'Sponsored'}
importances.index = [nice_names.get(i,i) for i in importances.index]
colors = [PURPLE if v > importances.values.mean() else MUTED for v in importances.values]
bars = ax2.barh(importances.index, importances.values, color=colors, height=0.6)
style_ax(ax2, 'Feature Importance')
ax2.set_xlabel('Importance Score', color=MUTED, fontsize=9)
ax2.grid(axis='y', alpha=0)
for bar, val in zip(bars, importances.values):
    ax2.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', color=MUTED, fontsize=8)

# 3. Avg views by category
ax3 = fig.add_subplot(gs[1, 0])
cat_views = df.groupby('category')['views'].median().sort_values(ascending=False)
bar_colors = [TEAL if v == cat_views.max() else PURPLE for v in cat_views.values]
ax3.bar(cat_views.index, cat_views.values/1000, color=bar_colors, width=0.6)
style_ax(ax3, 'Median Views by Category')
ax3.set_xlabel('Category', color=MUTED, fontsize=9)
ax3.set_ylabel('Views (K)', color=MUTED, fontsize=9)
plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=8)

# 4. Metrics card
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD_BG)
for sp in ax4.spines.values(): sp.set_color(GRID)
ax4.set_xticks([]); ax4.set_yticks([])
ax4.set_title('Model Performance', color=TEXT, fontsize=12, fontweight='bold', pad=10)
metrics = [('R² Score', f'{r2:.4f}', TEAL), ('MAE', f'{mae/1000:.1f}K views', GOLD), ('RMSE', f'{rmse/1000:.1f}K views', CORAL)]
for i,(lbl,val,col) in enumerate(metrics):
    y_pos = 0.72 - i*0.28
    ax4.text(0.5, y_pos+0.08, lbl, ha='center', va='center', color=MUTED, fontsize=11, transform=ax4.transAxes)
    ax4.text(0.5, y_pos-0.04, val, ha='center', va='center', color=col, fontsize=22, fontweight='bold', transform=ax4.transAxes)

fig.text(0.5, 0.97, 'Instagram Reels View Predictor — Model Results', ha='center', color=TEXT, fontsize=14, fontweight='bold')
plt.savefig('/home/claude/project/model_results.png', dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print("Plot saved.")
