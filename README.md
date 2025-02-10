# EPL-Player-Clustering
This project applies unsupervised clustering to normalised per-match performance metrics to group Premier League players into distinct clusters that reveal differences in playing styles and roles

# EPL Player Clustering

Hi there!

Welcome to my EPL Player Clustering project. In this project, I use unsupervised learning to group Premier League players based on their performance metrics. The goal is to see if we can uncover interesting patterns about players’ performances by clustering them according to per-match statistics.

## What’s This Project About?

- **Objective:**  
  Group players into clusters based on their per-match performance metrics (like goals, assists, passes, and tackles).

- **Key Features:**  
  - I’ve computed per-match rates for:  
    - Goals per match  
    - Assists per match  
    - Passes per match  
    - Tackles per match  
  This helps to normalize the data regardless of how many matches a player has played.

- **Clustering Techniques:**  
  I experimented with two clustering methods:
  - **KMeans:** I tried out cluster numbers from 2 to 5 and selected the best one using the silhouette score.
  - **Agglomerative Clustering:** I also compared this approach using the silhouette score to see if it provided different insights.

- **Validation:**  
  The silhouette score helps determine the optimal number of clusters and gives an idea of how well the data is grouped.

## The Dataset

For this project, I’m using Premier League player statistics. A good dataset to work with is available on Kaggle:
- [Premier League All Players Stats 23/24 on Kaggle](https://www.kaggle.com/datasets/orkunaktas/premier-league-all-players-stats-2324)


## How to Run It

1. **Download the dataset** and save it as `epl_player_stats.csv` in this folder.
2. Make sure you have Python 3 installed.
3. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn
4. Run the script
   `python player_clustering.py`
   
The script will show silhouette scores for different numbers of clusters, pick the best one, and print sample clustering results for a few players.

## What’s Inside the Code?

- Data Preparation: The script cleans the dataset, handles missing values, and calculates per-match performance metrics.
- Clustering: It applies KMeans (with silhouette score-based selection for the best number of clusters) and Agglomerative Clustering, comparing the two approaches.
- Results: You’ll see which players fall into each cluster, which can give you insights into different playing styles or roles.

## About Me

I’m Younus Emre, a data enthusiast who loves both football and machine learning. I created this project to dig deeper into player performance and uncover interesting patterns through clustering. I hope you find it as intriguing as I did!

## License

This project is for educational purposes. Feel free to use, modify, and share the code.
