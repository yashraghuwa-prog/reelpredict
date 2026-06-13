# 🎬 ReelPredict AI — Instagram Reels View Count Predictor

> Predict how many views your Instagram Reel will get — before you even post it.

## 🎯 Problem Statement
Content creators on Instagram struggle to know what will perform well before posting. This project builds an ML model that predicts the view count of an Instagram Reel based on content features, timing, and account history.

## 📊 Dataset
- **Type:** Synthetic dataset (mimicking real Instagram Reels behavior)
- **Size:** 5,000 records, 14 features
- **Target Variable:** View count (regression)

### Features Used
| Feature | Description |
|---|---|
| category | Content type (dance, comedy, food, etc.) |
| day_of_week | Day of posting |
| post_hour | Hour of posting (0–23) |
| followers | Account follower count |
| reel_duration_sec | Length of reel in seconds |
| num_hashtags | Number of hashtags used |
| caption_length | Length of caption in characters |
| has_music | Whether reel has background music |
| has_trending_audio | Whether reel uses trending audio |
| collab_post | Whether it's a collaboration |
| avg_past_views | Creator's average past views |
| past_engagement_rate | Historical likes/views ratio |
| num_mentions | Number of accounts mentioned |
| is_sponsored | Whether it's a paid promotion |

## 🤖 Model
- **Algorithm:** Gradient Boosting Regressor (scikit-learn)
- **Target Transform:** Log1p (to handle skewed view distribution)
- **Train/Test Split:** 80/20

## 📈 Results
| Metric | Score |
|---|---|
| R² Score | 0.7245 |
| MAE | ~9,130 views |
| RMSE | ~31,176 views |

## 🚀 Run the Demo
```bash
pip install streamlit scikit-learn pandas numpy matplotlib seaborn
streamlit run app.py
```

## 🗂️ Project Structure
```
├── instagram_reels_data.csv   # Generated dataset
├── generate_dataset.py        # Dataset generation script
├── train_model.py             # Model training + evaluation
├── model.pkl                  # Saved trained model
├── model_results.png          # Performance visualizations
├── app.py                     # Streamlit demo app
└── README.md
```

## 🛠️ Tech Stack
- Python 3.x
- scikit-learn (ML model)
- Pandas, NumPy (data processing)
- Matplotlib, Seaborn (visualization)
- Streamlit (demo app)

## 👤 Author
Built for Amazon ML Summer School Application
