#!/usr/bin/env python3
"""
Project 2: Player Performance Clustering

This script loads Premier League player statistics, computes per-match performance metrics,
and then applies clustering algorithms to group players. It compares different cluster counts using the silhouette score,
and also compares KMeans with Agglomerative Clustering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# -------------------------------
# 1. Data Loading and Preprocessing
# -------------------------------

# Load player statistics dataset
df = pd.read_csv('epl_player_stats.csv')

# Expected columns: 'Player', 'Team', 'Matches', 'Goals', 'Assists', 'Passes', 'Tackles'
# Fill in any missing values or handle divisions by zero
df['Matches'] = df['Matches'].replace(0, 1)  # Prevent division by zero

# -------------------------------
# 2. Feature Expansion: Compute Per-Match Performance Metrics
# -------------------------------

# Create new features that reflect performance per match
df['Goals_per_match'] = df['Goals'] / df['Matches']
df['Assists_per_match'] = df['Assists'] / df['Matches']
df['Passes_per_match'] = df['Passes'] / df['Matches']
df['Tackles_per_match'] = df['Tackles'] / df['Matches']

# Select the expanded feature set for clustering
features = ['Goals_per_match', 'Assists_per_match', 'Passes_per_match', 'Tackles_per_match']
X = df[features]

# Standardise the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. Clustering: Determine Optimal Number of Clusters with KMeans
# -------------------------------

# Try different numbers of clusters and evaluate using silhouette score
sil_scores = {}
for k in range(2, 6):  # Trying clusters from 2 to 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    sil_scores[k] = score
    print(f"KMeans with {k} clusters: Silhouette Score = {score:.3f}")

# Choose the number of clusters with the highest silhouette score
optimal_k = max(sil_scores, key=sil_scores.get)
print(f"\nOptimal number of clusters (KMeans): {optimal_k}")

# Run final KMeans clustering with optimal_k
final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster_KMeans'] = final_kmeans.fit_predict(X_scaled)

# -------------------------------
# 4. Model Comparison: Compare with Agglomerative Clustering
# -------------------------------

agg_cluster = AgglomerativeClustering(n_clusters=optimal_k)
agg_labels = agg_cluster.fit_predict(X_scaled)
agg_score = silhouette_score(X_scaled, agg_labels)
print(f"\nAgglomerative Clustering with {optimal_k} clusters: Silhouette Score = {agg_score:.3f}")

# Add Agglomerative labels to dataframe for comparison
df['Cluster_Agglomerative'] = agg_labels

# -------------------------------
# 5. Output Results
# -------------------------------

print("\nSample clustering results (first 10 players):")
print(df[['Player', 'Team', 'Cluster_KMeans', 'Cluster_Agglomerative'] + features].head(10))
