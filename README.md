# Movie Recommender System

A machine learning project that recommends movies using two approaches, content based filtering and collaborative filtering. Built with Python, scikit-learn, Surprise and Streamlit. 

---

## Overview

This project explores two core recommendation techniques on the MovieLens 1M dataset:
| Approach | Method | Input | Output |
|---|---|---|---|
| Content-Based | TF-IDF + Cosine Similarity | Movie title | Similar movies by genre |
| Collaborative | SVD Matrix Factorization | User ID | Personalized recommendations |

## Model Performance
| Model | RMSE | MAE |
|---|---|---|
| Baseline (random) | ~1.05 | ~0.84 |
| SVD Collaborative | 0.8729 | 0.6845 |

SVD improved RMSE by ~17% over the baseline.

---

## Tech Stack
- **pandas** - Data loading and manipulation
- **scikit-learn** - TF-IDF, cosine similarity
- **Surprise** - SVD collaborative filtering 
- **Streamlit** - interactive web app
- **Matplotlib/Seaborn** - data visualization

---

## Improvements to be made
- Hybrid model that combines both approaches
- Integrate TMDB API for movie posters and visuals

## Author
Ariel Xie