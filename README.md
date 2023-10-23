# Enhanced Movie Recommendations with Collaborative Topic Modeling

Welcome to this GitHub repository explaining my journey into building an end-to-end movie recommender system. Inspired by the captivating influence of Tik-Tok's personalized feed and a strong desire to learn how industry recommender systems are designed, I am on a mission to learn the inner workings of cutting-edge recommendation systems.

Just came here for a great movie recommendation - follow [this link](https://nmfasano5-content-based-movie-recommendation-system.hf.space) to get your recommendation now!

In this repo: 
- The 'database' folder contains the datasets used in this project, including the Python code used for web scraping, wrangling, and synthesizing the datasets.
- The 'external_packages' folder contains open source repositories that were cloned and modified to create the collaborative topic model.
- The 'movie_recsys' folder contains notebooks and Python scripts used to create the movie recommender system based on the collaborative topic model. 
- The 'model_testing' folder contains notebooks for prototyped models
- This README.md file details the full life-cycle of the deployed recommender system, especially the data engineering and machine learning pipelines.
____________________________________________________________________________________________

### Jump to section
* [Abstract](#abstract)
* [Introduction: project motivation and scope](#introduction-project-motivation-and-scope)
* [Datasets used in this work](#datasets-used-in-this-work)
* [Data wrangling and cleaning](#data-wrangling-and-cleaning)
* [Data preprocessing](#data-preprocessing)
* [Model training and evaluation](#model-training-and-evaluation)
   * [Latent Dirichlet Allocation (LDA)](#latent-dirichlet-allocation-lda)
   * [Collaborative Topic Model (CTM)](#collaborative-topic-model-ctm)
* [Model deployment](#model-deployment)
* [Conclusions and future work](#conclusions-and-future-work)
* [Resources](#resources)
 
____________________________________________________________________________________________

### Abstract

Movie recommender systems have become indispensable tools in the era of digital streaming, providing personalized
content suggestions to users who are frequently overwhelmed by the vast selection of movies and television shows
available to watch. In this work, we build a collaborative topic model for making movie recommendations. The
collaborative topic model is a hybrid approach that combines traditional matrix factorization with topic modeling of
movie film scripts. We show that this hybrid model achieves a modest (1%) improvement in root-mean-square error
compared to traditional matrix factorization approaches, but alleviates the item cold-start problem with a recall@10
score of 44% for movie titles not seen during training.

____________________________________________________________________________________________

### Introduction: project motivation and scope

Consumers who visit a streaming platform to find something to watch are faced with an exorbitant amount of high-quality movies and television shows, making finding the right film overwhelming. This frustration decreases user satisfaction and overall watch time, affecting the business revenue of the streaming platforms that provide the content. To address this issue, recommendation models have been built that attempt to filter out films that a user is unlikely to engage with. Two broad classes of methods that can be applied to this filtering problem are collaborative-based filtering and content-based filtering. In collaborative filtering, previous user-item interactions (e.g. ratings or clicks) are leveraged to recommend novel items to a user. A popular collaborative filtering model is matrix factorization, which was popularized during the 2006 Netflix prize competition [1]. In content-based filtering, item features (e.g. film text or other metadata) are used to recommend new items to a user who has interacted with similar items in the past. Collaborative filtering methods provide diverse recommendations to users but cannot make recommendations for new films for which there are no ratings (i.e. the item cold-start problem). On the other hand, content filtering methods can easily recommend new films but tend to lack diversity in recommendations. The complementary set of pros and cons for these two methods has prompted the development of hybrid models which combine the merits of both approaches.

In this work, we build a hybrid movie recommender system (FIG. 1) based on the collaborative topic model (CTM) [2]. In CTM, we combine singular value decomposition (SVD) for matrix factorization of the user-movie ratings matrix (i.e. a collaborative model) with Latent Dirichlet Allocation (LDA) for topic modeling of film scripts (i.e. a content model). The primary advantage of this hybrid approach is that it allows for in-matrix and out-of-matrix predictions of a user's rating for any movie in the database. In-matrix predictions provide ratings for movies that have been rated by at least one user, while out-of-matrix predictions provide ratings for movies that have never been rated which alleviates the item cold-start problem.

In the remainder of this README.md file, we first discuss the datasets used in this project and how they were cleaned and processed for training the machine learning algorithms. We then build the LDA model for topic modeling of the film script dataset and integrate this model into the SVD model to form the complete CTM model. Finally, the CTM and SVD models are evaluated by computing the root mean square error (RMSE) and recall@k metrics on a set of data not seen during training. We conclude by discussing possible extensions to this hybrid approach such as using word embeddings instead of topic modeling for developing the content side of the algorithm.

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/readme_images/movie_rec_pipeline.png" width="100%">
</picture>

*FIG. 1: Workflow diagram of the recommender system, highlighting the data engineering and machine learning pipelines.* 
____________________________________________________________________________________________

### Datasets used in this work
The following datasets were used in this work:
1. Film scripts 
	- Source: Web scraped from IMSDb.com and springfieldspringfield.co.uk
	- Content: film title, film release year, film script for 35,000+ movies and 130,000+ TV episodes
	- Note: The code for web scraping the film scripts is available on this project's GitHub repository [3].
2. IMDb Non-Commercial Datasets
	- Source: developer.imdb.com/non-commercial-datasets
	- Content: film title, average rating, genres, runtime, cast, and crew
	- Note: This dataset is freely available for noncommercial use
3. The Movie DataBase (TMBD)
	- Source: themoviedb.org
	- Content: movie metadata (budget, revenue, runtime, cast, release date, etc.), movie posters
	- Note: This dataset is freely available for noncommercial use. In this work, the TMDB dataset was primarily used for displaying poster art on the movie recommender web app [4].
4. MovieLens Dataset
	- Source: grouplens.org
	- Content: 25+ million movie ratings using a 5-star rating system. The dataset contains 283228 users and 58098 movies and all users rated at least one movie 
	- Note: This dataset is freely available for noncommercial use
____________________________________________________________________________________________
### Data wrangling and cleaning
To wrangle these datasets into one central database, we needed to find matching entry IDs between the four datasets. Conveniently, the MovieLens dataset contains matching IDs for the TMDB and IMDb datasets, so I only needed to focus on finding a matching ID between the film scripts dataset and one of the other three datasets. A few different approaches were taken to systematically find these IDs, such as matching movie titles and movie years between the script dataset and IMDb/TMDB/MovieLens datasets. Additionally, the themoviedb.org website provides an API that can be queried using movie title and year to return the corresponding IMDb and TMDB IDs. Any remaining movies from the film script dataset that did not have a matching IMDb ID were filled in manually.

Finally, it was necessary to clean the datasets for training the machine learning models so that the data could be available as metadata for the recommender app. Some examples of the data issues that were cleaned are as follows:

- Fixed errors -- e.g. changed movie year from "0000" to "1999" or from "20147" to "2017"
- Put titles into consistent formatting across datasets -- e.g. changed movie titles in script dataset from "Dark Knight, The" to "The Dark Knight."
- Removed duplicate entries from all datasets
- Removed data entries that only contained irrelevant script text such as "None" or "More Movie Scripts | Request a Movie Transcript"

____________________________________________________________________________________________

### Data preprocessing
The ratings matrix obtained from MovieLens is already in a form ready to be used for model training, so I only needed to focus on the film scripts dataset. To process the script data for the LDA model, the following Natural Language Processing tasks were performed:

- made the entire corpus lowercase and removed all punctuation
- stop word removal using NLTK's list of stop words
- word mapping for words with the same meaning but different spellings (e.g. okay $\rightarrow$ ok)
- implemented the option for stemming and lemmatizing the corpus
- created a bag of words representation using the top 10,000 words in the vocabulary which occurred at least three times but in no more than 90% of the documents.

FIG. 2 shows the top 60 words and their word count across the entire corpus after the removal of stop words and words present in greater than 90% of the documents. For this figure, lemmatization was turned off during preprocessing which is why the words "friend" and "friends" appear. Here we see the diversity of words present in the corpus where the topic of family seems especially prevalent (e.g. words like mom, mother, baby, son, wife, father, and brother).

<picture>
	<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/eda_scripts_out/word_count_top_100.png" width="100%">
</picture>

*FIG. 2: Word count for the top 60 words present in the vocabulary after data preprocessing.* 
____________________________________________________________________________________________

### Model training and evaluation
In this section, we discuss the details on creating the Collaborative Topic Model (CTM). First we discuss Latent Dirichlet Allocation for topic modeling of the film scripts dataset and then show how to integrate the learned topics into the matrix factorization framework, creating the final CTM model. The CTM model is then trained on training data and evaluated on held out test dataset using disparate metrics (RMSE and recall@k).

____________________________________________________________________________________________

#### Latent Dirichlet Allocation (LDA)
Latent Dirichlet Allocation (LDA) is a three-level hierarchical Bayesian model widely used for uncovering latent topics within a large corpus of documents [5]. In LDA, each document is represented as a mixture of topics and the topics themselves are represented by a mixture of words. Importantly, LDA allows for documents to be represented by a misture of topics which will allow the model to distinguish between movies that have the same dominant topic but very different sub-topics. For example, consider the movies 'Little Giants' and 'Remember the Titans.' Both films are predominantly about sports, but 'Little Giants' is also a comedy film geared toward younger audiences, and 'Remember the Titans' is also a biographical film about the end of segregation in American schools. We wish for LDA to distinguish between these sub-topics when making recommendations.


<picture>
	<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/model_eval_out/lda_training.png" width="100%">
</picture>

*FIG. 3: (a) Evaluated perplexity on the training data and testing data as a function of the number of topics in the LDA model ($n_{components}$). (b) The top 15 words in the first five topics of the LDA model where $n_{components}$ = 20.* 

We used scikit-learn's implementation of LDA for model training [6]. To determine the number of topics, we ran 5-fold cross-validation on 10,000 randomly selected movie scripts where each fold had 8000 scripts in the train set and 2000 scripts in the test set. Here we use perplexity as the evaluation criteria, where perplexity computes the normalized log-likelihood on a set of documents and roughly represents how surprised the model is to see this set of documents based on the set of documents it was trained on. In FIG. 3a we plot the average perplexity computed from the cross validation procedure as a function of the number of topics ($n_{components}$) for both the training and testing datasets. The figure indicates that the reduction in perplexity on the test set with an increasing number of topics plateaus at $n_{components}$ $\approx$ 20. 

FIG. 3b shows the top words present in the first five topics of the 20-topic model. Most of these topics are reasonably interpretable. Loosely speaking, topic 1 represents war-type movies, topic 2 represents Christmas/holiday movies, topic 4 represents crime-type movies, and topic 5 represents sport-type movies. Topic 3 is a bit harder to label, but seems to represent a class of people, particularly when addressed formerly with words such as mr, mrs, dear, and darling. Each topic, however, does have some words that seem misplaced, such as harassment in topic 2 and the words nick, gwen, gandhi, jasper, and phil in topic 5.

<picture>
	<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/model_eval_out/lda_eval.png" width="100%">
</picture>

*FIG. 4: Distribution of topics for individual films and the top 10 words present in the top two topics for each movie. The selected movies are (a) The Dark Knight, (b) A Christmas Carol, (c) Remember the Titans, and (d) Schindler's List.* 

With the latent topics now learned, we can represent each film script in the corpus as a distribution over these topics. In FIG. 4, we plot these distributions for the selected movies (a) The Dark Knight, (b) A Christmas Carol, (c) Remember the Titans, and (d) Schindler's List. Each subplot (a-d) shows the distribution of topics and the top ten words in the top two topics for that movie. Note that each movie is primarily represented by only 2-3 topics and the majority of the remaining topic proportions are less than 5%. 

Now that we have computed the distribution of topics for each movie, we can find the top n most similar films for any particular movie. To do that, we compute the cosine similarity between an input movie's topic vector with the topic vector of all other movies in the database, ranking the values in descending order. In FIG. 5, we show the results of this ranking procedure for the input movies "Remember the Titans" and "Little Giants." Notice how the model recommends serious sports films for "Remember the Titans" and more family/comedy-style sports movies for "Little Giants". Note that the model does not see the genre of the movie at training time.

<picture>
	<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/readme_images/example_recs_little_giants_and_remember_the_titans.png" width="100%">
</picture>

*FIG. 5: (left) Top five most similar movies to "Remember the Titans." (right) Top five most similar movies to "Little Giants."* 



The LDA model does have some drawbacks in practice. One notable drawback is that the LDA model requires one to know the number of topics apriori. In this work, we used cross-validation to select the number of topics with perplexity as the evaluation criteria. In principle, one can use a downstream (business-oriented) task to fix the number of topics. Another drawback of the LDA model is that the model can produce topics with words that are uncorrelated, especially in noisy datasets. As pointed out in FIG. 3, the LDA model learned here exhibits this behavior to some extent. The issue can be avoided by increasing the amount of training data or by using a coherence metric for selecting the number of topics [7]. 

A final drawback of the LDA model is that the topics are static and do not capture any time evolution. In our dataset, this becomes apparent when we ask for the most similar movies to an input movie that takes place in the past century but was filmed in the current century. In FIG. 5, we show the ranking of most similar movies for the input movies "The Help" and "Green Book." For both movies, the model has done a reasonable job finding movies with similar content, but these movies are not likely to be the most relevant recommendations since all of the movies were filmed before 2000. For "Green Book," 3/5 of the most similar films were made in the 1940s. One way to circumvent this problem would be to use an extension to the LDA model, known as Dynamic LDA, that attempts to model the evolution of topics over time to account for how the usage of certain words has evolved [8]. Another way to address this problem would be to devise a reordering method that is based on some business logic.


<picture>
	<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/readme_images/example_recs_the_help_and_green_book.png" width="100%">
</picture>

*FIG. 6: (left) Top five most similar movies to "The Help." (right) Top five most similar movies to "Green Book."* 

____________________________________________________________________________________________

#### Collaborative Topic Model (CTM)
In the previous section, we built a topic model and found that it can provide reasonable movie recommendations based purely on the movie's content. In this section, we integrate the LDA model into the SVD model forming the complete CTM model.

The SVD model begins with the ratings matrix, $R \in \mathbb R^{M\times N}$, which contains the ratings for $M$ movies and $N$ users and each element $r_{mn}$ contains user $n$'s rating of the $m^{th}$ movie. In general, this ratings matrix is extremely sparse ($>$99% sparsity for the MovieLens dataset) since the majority of users only rate a few items. To address this problem, SVD factorizes the ratings matrix into the product $Q^TP$ where $P \in \mathbb R^{d\times N}$ is a matrix of latent vectors for representing the users, $Q \in \mathbb R^{d\times  M}$ is a matrix containing the latent vectors for representing the items, and d is the size of these latent vectors. The power of matrix factorization is that, even though the original rating matrix is large and sparse, the product $Q^TP$ is completely dense and is guaranteed to be the best rank-d approximation to $R$ under the Frobenius norm.

To incorporate the topic vectors from LDA into this SVD framework and form the CTM model, we replace $Q$ with $Q - \Theta$, where $\Theta \in \mathbb R^{dxM}$ contains the latent topic vectors for all $M$ films in the dataset. Note that we consider the individual topics $\theta$ to be fixed from the LDA model of the previous section, but they can be learned simultaneously with $Q$ and $P$ by employing an EM-style algorithm [2]. However, it has been shown empirically that fixing $\Theta$ from plain vanilla LDA yields similar results and saves computational time. 

To train the CTM model, we modified the SVD algorithm in the Surprise library to take in the topic matrix $\Theta$ as an input. There are several hyperparameters to learn in the CTM model, including the number of components ($n_{components}$) and regularization parameters for the user matrix ($\lambda_p$), the item matrix ($\lambda_q$), and the topic matrix ($\lambda_{\theta}$). Note that traditional matrix factorization can be recovered by setting $\lambda_{\theta}$ = 0. Here we fix the number of components according to the number of topics from the LDA model ($n_{components}$ = 20). To learn the remaining hyperparameters, we perform a grid search using 5-fold cross-validation. The results of this grid search are summarized in TABLE 1. Using these optimal parameters, we then trained a CTM and SVD model on the full training set and evaluated the root-mean-square-error (RMSE) and recall@k metrics on a held-out test set. 

The results of these evaluation metrics are summarized in TABLE 1, where we see that CTM has a modest (1%) improvement in RMSE compared the SVD model, a 0.4% improvement in recall@10 score compared to the SVD model, and achieves a 43.5% recall@10 score for out-of-matrix predictions that only the CTM model can make. Finally, in FIG. 7, we plot the recall@k evaluation metric as a function of k for in-matrix predictions of the SVD and CTM models (open red circles and filled blue circles, respectively) and for out-of-matrix predictions for the CTM model (purple filled circles). Out-of-matrix predictions are not possible for the SVD model. For in-matrix predictions, both models reach an upper recall limit at k$\approx$40 of 0.61. For out-of-matrix predicitons, the CTM model reaches an upper recall limit at k$\approx$50 of 0.54.



<picture>
	<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/ctm_model_eval/eval_metrics.png" width="100%">
</picture>

*TABLE 1: Comparison of hyperparameters and evaluation metrics for SVD and CTM models.* 

_

<picture>
	<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/ctm_model_eval/recall_at_k.png" width="60%">
</picture>

*FIG. 7: Recall@k vs k for in-matrix predictions (SVD and CTM models) and for out-of-matrix predictions (CTM model).* 

____________________________________________________________________________________________

### Model deployment
To deploy the developed model, we built a web-based app using [gradio](https://www.gradio.app/) and then hosted the app on [Hugging Face Spaces](https://huggingface.co/spaces), which allows the application to be available permanently and for free.

The code contains the following features:

1. The gradio code used to create the app. I chose to utilize gradio's low-level API ([Blocks](https://www.gradio.app/guides/quickstart#blocks-more-flexibility-and-control)) for building the app since it allows for more flexible layouts and data flows). 
2. A movie recommendation function (def movie_rec) that takes in a movie, some filters, and then returns the most similar movies (plus that movie's metadata) from the cosine-similarity matrix after filtering out low-ranking and Adult films. 
3. A Query parser (def update_radio(text)) that takes in the text from the search box and returns the most relevant movies from the dataset. The parser is case-insensitive and robust to punctuation and extraneous spaces, but it will not handle spelling mistakes.

The deployed recommender can be tested by following [this link](https://nmfasano5-content-based-movie-recommendation-system.hf.space). A screenshot showing some example recommendations based on the movie input "Remember the Titans" is shown in the following figure. 

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/readme_images/example_recsys_input.png" width="100%">
</picture>
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_recsys/readme_images/example_recsys_output.png" width="100%">
</picture>

*FIG. 8: Snapshot of the movie recommender web app deployed to Hugging Face's spaces. In this example, the user began searching for a movie and then selected "Remember the Titans" from the list of available movies. The user chooses various filters to be applied to the recommendations and clicks Recommend. At this point, 5 movie titles along with some metadata and IMDb.com links to the movies' title pages are shown.* 

____________________________________________________________________________________________

### Conclusions and future work
In this work we developed a collaborative topic model (CTM) for movie recommendations to users, combining Latent Dirichlet Allocation (LDA) applied to a corpus of film scripts and Singular Value Decomposition (SVD) applied to a user-movie ratings matrix. We find a 1% improvement in RMSE from the CTM model compared to SVD model and compute a recall@10 value of 44% for out-of-matrix predictions which alleviates the item cold-start problem. With the addition of movie content information (represented in this work as topics from LDA model) providing superior rating predictions compared to matrix factorization approaches alone, we plan to experiment with word embeddings to represent the film script corpus which may allow for more flexible representations of the corpus and lead to even better performance.

____________________________________________________________________________________________

### Resources
[1] J. Bennett, S. Lanning, et al., in Proceedings of KDD cup and workshop, Vol. 2007 (New York, 2007).

[2] C. Wang and D. M. Blei, in Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and
data mining (2011).

[3] [https://github.com/nfasano/movie_recsys/](https://github.com/nfasano/movie_recsys/)

[4] [https://huggingface.co/spaces/nmfasano5/content_based_movie_recommendation_system/](https://huggingface.co/spaces/nmfasano5/content_based_movie_recommendation_system/)

[5] D. M. Blei, A. Y. Ng, and M. I. Jordan, Journal of machine Learning research 3, 993 (2003).

[6] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss,
V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, Journal of Machine
Learning Research 12, 2825 (2011).

[7] S. Syed and M. Spruit, in 2017 IEEE International conference on data science and advanced analytics (DSAA) (IEEE,
2017).

[8] D. M. Blei and J. D. Lafferty, in Proceedings of the 23rd international conference on Machine learning (2006) pp. 113–120.

Other useful sites and blog posts:

[a] [Microsoft Recommenders](https://github.com/recommenders-team/recommenders) - well-maintained GitHub repository detailing the best practices for building and deploying recommender systems.

[b] [Nvida](https://docs.nvidia.com/deeplearning/performance/recsys-best-practices/index.html) - Document discussing best practices for building recommender systems

[c] [Example of collaborative Topic model used at NYT](https://archive.nytimes.com/open.blogs.nytimes.com/2015/08/11/building-the-next-new-york-times-recommendation-engine/?mcubz=0&_r=0) - Discusses the NYTs experimentation with collaborative topic models (LDA is used for topic model) in production.



