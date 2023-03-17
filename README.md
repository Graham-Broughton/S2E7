# Kaggle Tabular Playground Series - July 2022
## Clustering (placed 42/1253)

# Table of Contents

1. [Overview](https://github.com/Graham-Broughton/Kaggle/blob/master/Clustering_072022/README.md#Overview)
2. [Exploratory Data Analysis](https://github.com/Graham-Broughton/Kaggle/blob/master/Clustering_072022/README.md#Exploratory-Data-Analysis)
4. [Feature Engineering](https://github.com/Graham-Broughton/Kaggle/blob/master/Clustering_072022/README.md#Feature-Engineering)
5. [Modelling](https://github.com/Graham-Broughton/Kaggle/blob/master/Clustering_072022/README.md#Modelling)
6. [Conclusion](https://github.com/Graham-Broughton/Kaggle/blob/master/Clustering_072022/README.md#Conclusion)

# Overview
This month's Kaggle TPS challenge was all about clustering. Clustering is a type of unsupervised learning (the data does not have a target/feature to predict) where the model attepts to label each datapoint in such a way that data belonging to the same label share some form of similarity to one another, or they are more dissimilar from other clusters than they are from eachother. Clustering is an important branch of machine learning that is utilized by insurance companies to detect fraud, in Biology to classifying the natural world and in marketing to discover customer segments to optimize marketing programs, these are just a few there ar emany more uses for clustering. My workflow for this challenge was: perform extensive EDA, feature engineering, clustering models, using the clustering models predictions to train classifiers then finally creating a plethora of different ensembles. Possibly the most challenging (and frustrating) aspect of this competition was the absence of a reliable scoring metric we could use for cross validation. The evaluation metric used was 'Adjusted Rand Score', which basically measures how similar the predicted clusters are to the true clusters, heavily penalizing random labels. All other methods for scoring clusters (silhouette, calinski harabasz or davies bouldin score) did not show any correlation with the adjusted Rand Score.

# Exploratory Data Analysis

## Basic Analysis
The dataset had 98000 observations and 29 features - seven integer columns seeming to represent categorical or discrete data and the rest floats. The float features were also made up of two groups: one group was standardized (mean=0, std=1) the other was not. Plotting the floats on histograms and boxplots provided strong support that the standardized group followed a gaussian distribution while the non-standardized appeared 'close' to gaussian. The boxplots highlighted the large number of datapoints outside of the IQR - outliers - to reduce the impact of outliers I will try different methods of scaling. The bar charts of the integer features showed truncated, skewed right, gaussian-like distributions, and even more outliers in thier boxplots as a result of the skewed distributions. Power transforming these features will reduce truncation and skew, changing the distribution to more gaussian. A correlation heat map showed inter and intra specific relationships in the integer and non-standardized float features, good news if it was a supervised learning problem but I am not sure how it will effect clustering. Finally, I used the Shapiro-Wilkinson test for normality on all features which concluded that only the standardized float features follow a normal distribution eg. power transform all other features to make them more normal. Here is an example of the histograms, barplots and boxplots from each data group.

<table border="0">
  <tr style="width:100%">
    <th>Standardized Floats</th>
    <th>Integers</th>
    <th>Non-Standardized Floats</th>
  </tr>
  <tr style="width:100%">
    <td> <img src="/Clustering_072022/src/images/f0_box.png"  alt="1" width="100%" height = 246px > </td>
    <td> <img src="/Clustering_072022/src/images/f11_boxplots.png"  alt="1" width="100%" height = 246px > </td>
    <td> <img src="/Clustering_072022/src/images/f25_box.png"  alt="1" width="100%" height = 246px > </td>
  </tr>
  <tr style="width:100%">
    <td> <img src="/Clustering_072022/src/images/f0_float.png"  alt="1" width="100%" height = 246px > </td>
    <td> <img src="/Clustering_072022/src/images/int_bars.png"  alt="1" width="100%" height = 246px > </td>
    <td> <img src="/Clustering_072022/src/images/f25_float.png"  alt="1" width="100%" height = 246px > </td>
  </tr>
</table>

## Number of Clusters
I used three methods to deduce the number of clusters: Principle Component Analysis (PCA) explained variance, K-means and Gaussian Mixture Models. 

### Principle Component Analysis
PCA is a common dimensionality reduction technique where 'n' principle components replace the existing feature space. It is not a clustering algorithm but my anecdotal experience has shown it can provide additional evidence towards the number of clusters. Principle components are essentially just straight lines which pass through the origin. They are iteritively fit to the data in such a way where the first principle component maximizes the variance along its length and each following component is placed orthogonal to the one before it, while still maximizing variation slong its length. Placing them in this manner ensures they are not correlated with eachother (orthogonal) and captures as much information from the data in th fewest cmoponents ossible. When you plot the cumulitive explained variance per component, the component number where the explained variance slows down is indicative of the number of clusters. Here, we can see the gain in explained variance drop off sharply after component number seven. If the other methods do not produce a clear answer, PCA explained varince can help chose a winner.

<p align="center" width="100%">
    <img width="40%" src="/Clustering_072022/src/images/PCA_explained_variance.png"> 
</p>

### K-Means
K-Means is a clustering algorithm that aims to partition the data into 'n' clusters using the means of the cluster's datapoints, or centroids as they are called here. It iterates through a two stage cycle - assignment and update - until a specified convergence threshold ahs been achieved. The assignment step consists of assigning each datapoint to the centroid that it is closest to. The update step then creates new centroids using the cluster labels that were just defined. Since K-Means requires the number of clusters prior to clustering, a range of the potential number of clusters of is passed through K-Means to determine the most likely fit using various metrics. I used Silhouette score, Davies-Bouldin and Calinski Harahasz scores to create the plot below, inertia is probably the most common but it was not useful in this instance.
![KMeans scores](/Clustering_072022/src/images/KMeans_scores_raw.png)
Focusing on the silhouette score, we can see the largest decrease is from 6 to 7 components but as low as 5 make sense using Davies-Bouldin scoring. To understand what the clusters actually looked like, I also plotted visualizations of them in 2D space via PCA dimensionality reduction.
![KMeans PCA](/Clustering_072022/src/images/kmeans_clusters_raw.png)
Surprisingly, the raw dataset produced the best scores. One of the caveats of using K-Means is that it produces spherical clusters and does not perform well on dense or bloblike datasets. We can see in the 2D plots that the data is both dense and blob-like with scaling/transforming it making it even more dense and blob-like. So, I decided to try a clustering model more robust to dense, overlapping datasets.

### Gaussian Mixture Model
Gaussian Mixture Models (GMM) use 'n' gaussian functions to compute the probability of each datapoint belonging to each cluster. Since the parameters of the gaussian functions are unknown, the Expectation-Maximization algorithm is used to iteratively predict the likelyhood of cluster probabilities then use these probabilities to estimate the gaussian parameters, repeat. Without using the EM algorithm is would be impossible to find cluster labels this way due to the latent variables present. GMM are better for dense, overlapping blob-like data because the probabilistic nature of them allows for uncertainty and a fluid seperation of clusters instead of the hard assignment like K-Means. I used three metrics for scoring the number of clusters: Bayesian Information Criteria, Akiake Information Criteria and Davies-Bouldin Score and plotted the results along with 2D visualizations of the clusters.
![Gaussian scores](/Clustering_072022/src/images/gaussian_scores_raw.png)
![Gaussian clusters](/Clustering_072022/src/images/gaussian_clusters_raw.png)
I would say that is very strong evidence of 7 clusters present in the data. Similar to K-Means, the raw data produced better results but even more noticable in this model. I think I will use a different metric instead of David-Bouldin going forward as it was very different from the others.

# Feature Engineering
## Dropped Features
There were ~15 or so of the standardized float features which I wanted to gauge their importance towards seperating clusters (along with all other features, but specifically these ones). After running a Bayesian Mixture Model, similar to Gaussian but it produced better results, the kernel desntity of each cluster label for each feature was plotted. Below is a representative sample of each data group.
![Target Distribution](/Clustering_072022/src/images/download.png)
We can see that there is next to no variation between the clusters in the standardized group ie. they are not contributing to the differentiation of cluster labels. When dropped from the data, the scoring metric went up considerably. The integer features had more varation than the non-standardized floats by a large margin. Now to find the best scaler/transformer.

## Scaling/Transformation/Feature Importance
I decided to experiment with different combinations of scaler and transformers to see if there was a mor eeffective combination than power transformer/robust scaler. I used boxplots to roughly evaluate these combinations based on how well they dealt with outliers and the general shape of the data, seen below. 
![power transformer boxplot](/Clustering_072022/src/images/pt_box.png)
The most promising or just interesting combinations were sent in for scoring and the results saved in 2.3.Feature_Engineering-Scaler&Transformer_Combinations. Interestingly, using just transformed integer values gave a decent result, 0.37, while using just the non-standardized floats produced 0.05. When scored together they reached ~0.6 so there is some covariance between these groups, probably what we were seeing the the correlation matrix. Also, the best combination was to just use power transformer on all features, I guess the outliers did not negatively affect the clustering.

# Modelling
My general idea for modelling past the original clusters was to select high confidence data points to use to train classifiers. I accomplished this by setting a threshold probability and filtering for data points having a higher probabilty than the threshold. I would use a threshold between 0.75 and 0.90, the higher threshold for sklearn's semi supervised label spreader which does not need much data to train on. I proceeded to create an ensemble out of BGMM's but made a grave error and overlooked the fact that each round of clustering has no guarantee that the clusters are labelled the same. I switched back to single model clustering but eventually thought of saving the initial centroid means, then iteratively saving the argmin of the vector norm of initial mean minus each new mean. This gives us the cluster label the new means are closest to. After days and days of running models my score increased by maybe a paltry 0.03, although I did come across an interesting package from scipy, dual annealing, which optimizes weight choice given classifier probabilities, weight bounds, and a function to optimize. The next breakthrough was SK-lego's Bayesian Gaussian Mixture Model Classifier, which creates a mixture model for each class to predict labels from. It was very effective, intuitively this means that our 7 classes are each made of their own gaussian mixtures, I went from 0.63 to 0.74. Now I went deep into ensemble mode (notebooks starting with 10.) trying out everything I could think of ending up using the classifier in the weighted soft voting and then to iteratively update the prediction after training on a portion of the data. I even created an assortment of visualizations where I would choose ensembles that either reduced or increased the second highest probabilities thinking they would be more confident or an artefeact from label changing, or ensembles that changed the target counts but not drastically. Here are the plots from my best submission out of these ensembles:
Before iterating over predictions:

![preupdate class counts](/Clustering_072022/src/images/Ensemble%234%232%231-preupdate%20Best%20Class%20Value%20Counts.png)
![preupdate second highest](/Clustering_072022/src/images/Ensemble%234%232%231%20Second%20Highest%20Prediction%20Probabilities%20After%20Classification.png)
After iterating over predictions:

![postupdate class counts](/Clustering_072022/src/images/Ensemble%234%232%231-postupdate-%20Prediction%20Counts.png)
![postupdate second highest](/Clustering_072022/src/images/Ensemble%234%232%231-Postupdate-%20Second%20Highest%20Prediction%20Probabilities%20After%20Classification.png)

Unfortunately, none of these methods were very effective at predicting an increase in competition metric. Frustratingly, getting a lucky initialization seemed to be the most important factor. With only days left, I made my final and best ensemble using weighted soft voting of the best few previous ensemble's Bayesian Gaussian Mixture Model Classifier probabilities. After tinkering with the weights I increased my score ~0.01 which translated to moving from ~100 to 42th place. Heres a visualization of how small the change in distributions between the original best ensemble and the new predictions.
![sub prime viz](/Clustering_072022/src/images/newplot.png)

# Conclusion
Overall, I enjoyed this competition and learned clustering in much finer detail than I had before. This was also my first time creating ensembles, so that was a great experience. I still have a hard time forcing my brain to accept that an ensemble of lower accuracy models can perform better than an ensemble of higher accuracy models. One objective I had for this challenge was to expand my visualizations, specifically using subplots as I was not very comfortable with them previously. Now I have no problems subplotting and can even create custom color palettes to make the plots more visually appealing and, more importantly, easy for my colour blind eyes to analyze. I implemented everything that I could think of for this challenge, even if I had more time I don't think I would be able to improve my score very much. I am happy with placing 42th considering how many competitors there were.



