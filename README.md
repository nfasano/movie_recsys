# Content-based movie recommendation system

Welcome to this GitHub repository explaining my journey into building an end-to-end movie recommender system. 

Just came here for a great movie recommendation - then follow [this link](https://nmfasano5-content-based-movie-recommendation-system.hf.space) to get your recommendation now.

In this repo:
- This README.md file details the full life-cycle of the deployed recommender system, especially the data engineering and machine learning pipelines.
- The database folder contains the datasets used in this project, including the Python code used for websraping, wrangling, and synthesizing the datasets.
- The 'recsys_content_based' folder contains notebooks and Python scripts used to create a content-based movie recommender. 
- The 'recsys_collab_based' folder (ongoing work) contains notebooks implementing collaborative-based recommender algorithms with the idea of eventually creating a hybrid model (collaborative topic model).


____________________________________________________________________________________________

### Jump to section
* [Introduction: project motivation and scope](#introduction-project-motivation-and-scope)      
* [Components of this recommender system](#components-of-this-recommender-system)   
    * [Data engineering pipeline](#data-engineering-pipeline)
       * [Curated datasets](#curated-datasets)
       * [Data wrangling](#data-wrangling)
       * [Database updating](#database-updating) 
    * [Machine learning pipeline](#machine-learning-pipeline)
       * [Data preprocessing](#data-preprocessing)
       * [Model building](#model-building)
       * [Model evaluation](#model-evaluation) 
    * [Model deployment](#model-deployment)
    * [Model feedback and future work](#model-feedback-and-future-work)
    * [Resources](#resources)
 
____________________________________________________________________________________________

## Introduction: project motivation and scope
The goal of this project was to build a movie recommender that suggests new movies to watch that are similar in content to an input movie. It helps users find unknown movies that have similar topics to their favorite movies. Latent Dirichlet allocation (LDA) was used to find the latent topics of the entire corpus and cosine-similarity was used to find pairs of movies with similar latent topics. The film script dataset was web-scraped from various sites and combined with additional metadata from publically available datasets (IMDb.com, tmdb.org, and MovieLens.com). 

The deployed recommender can be tested by following [this link](https://nmfasano5-content-based-movie-recommendation-system.hf.space). A screenshot showing some example recommendations based on the movie input "Remember the Titans" is shown in the following figure. The remainder of this README.md file details the workflow used to build the recommender system, including the data engineering and machine learning loops which allow for seamless improvements to any part of the system.

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/readme_images/example_recsys_input.png" width="100%">
</picture>
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/readme_images/example_recsys_output.png" width="100%">
</picture>

*Figure 1: Snapshot of the movie recommender web app deployed to Hugging Face's spaces. In this example, the user began searching for a movie and then selected "Remember the Titans" from the list of available movies. The user chooses various filters to be applied to the recommendations and clicks Recommend. At this point, 5 movie titles along with some metadata and IMDb.com links to the movies' title pages are shown.* 

____________________________________________________________________________________________

## Components of this recommender system
The following figure shows the life-cycle of this movie recommendation project, highlighting the data engineering and machine learning loops used to continuously make improvements to the recommender system. 
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/readme_images/movie_rec_pipeline.jpg" width="100%">
</picture>

*Figure 2: Life-cycle diagram for this recommender system highlighting the data engineering and machine learning loops, as well as model deployment and monitoring.* 

The diagram begins with data engineering, where we first collected disparate datasets through web scrapping of movie script data and downloading of public datasets available for noncommercial use. These datasets are then cleaned and synthesized into a coherent database by matching the movie IDs between datasets. Throughout the project, the dataset is continuously updated to fix errors or add missing data. This improved data quality leads to an improved model and a better experience for the user of the recommendation system. With a cleaned and synthesized dataset, the machine learning loop can begin. In this phase of the project, we first began by doing exploratory data analysis to understand the richness of the dataset and then preprocessed the dataset (created a bag of words representation) into a form ready to be used for training of the model. The model we chose is Latent Dirichlet Allocation (LDA) which models each document as a mixture of topics with each topic containing a mixture of words. The advantages and shortcomings of LDA are highlighted in the model evaluation phase. Finally, the model is deployed as a web-based app using gradio to build the app and Hugging Face Spaces to host the app permanently and for free.

The specifics of each phase are discussed in more detail in the following sections. The corresponding notebook or Python file is also highlighted if the reader is interested in looking into the source code, most of which is well documented. 

____________________________________________________________________________________________

### Data engineering pipeline

#### Curated datasets:
1) Movie and television scripts from IMSDb and Springfield! Springfield!
    - Source: https://imsdb.com/ and https://www.springfieldspringfield.co.uk/
    - Content: film title, year, film script for 35,000+ movies and 130,000+ TV episodes 
    - Note: This dataset was webscraped. See [dataset_film_scripts](https://github.com/nfasano/movie_recsys/tree/main/database/dataset_film_scripts) folder in this repo for the notebooks used to scrape the scripts.
2) IMDb dataset
    - Source: [IMDb dataset](https://www.imdb.com/interfaces/)
    - Content: film title, average rating, genres, runtime, cast/crew 
    - Note: This dataset is freely available for noncommercial use.
3) The Movie DataBase (TMBD)
    - Source: [themoviedb.org](https://www.themoviedb.org/?language=en-US)
    - Note: This dataset is freely available for noncommercial use. It was primarily used for adding poster art to the recommender app and for matching IDs between the scripts dataset and the IMDb dataset.
4) MovieLens Dataset
    - Source: [MovieLens.com](https://movielens.org/home)
    - Content: user_id, item_id, rating, timestamp for over 20 Million ratings provided by real users on the MovieLens website. Also provides the IDs for IMDb and TMDB for all movies in their database, making it trivial to combine this dataset with IMDb and TMDB metadata.
    - Note: This dataset is freely available for noncommercial use.

#### Data wrangling
(see [data_wrangling.ipynb](https://github.com/nfasano/movie_recsys/blob/main/database/dataset_film_scripts/data_wrangling.ipynb) notebook for implementation)
To make this data usable to train our machine learning models and be available as metadata for the recommender app, it is necessary to clean and synthesize the datasets into one database. To clean the web scraped data, we manually inspected the movie titles and year of release and fixed any inconsistencies or errors, and changed the movie title formatting to be the same as the IMDb and tmdb datasets. Some examples of this data cleaning are shown below:

1) Fixed errors -- e.g. changed movie year from "0000" to "1999" or from "20147" to "2017"
2) Put titles into consistent formatting across datasets -- e.g. changed movie_titles in script dataset from "Dark Knight, The" to "The Dark Knight" or from "Beautiful Mind, A" to "A Beautiful Mind".
3) Removed duplicate entries with identical (or nearly identical) scripts
4) Removed data entries that only contained irrelevant script_text such as "None" or "More Movie Scripts | Request a Movie Transcript"

The next step was to find matching title IDs from the web scrapped script dataset with the IMDb and tmdb datasets. A few different approaches were taken to systematically find these IDs, such as matching movie titles and movie years between the script dataset and IMDb/tmdb datasets. Additionally, the tmdb.org website provides a convenient API that can be queried for all data using either tmdb ID, IMDb ID, or movie title and year, which allows for finding all relevant movie data when given any one of these three. This API was particularly helpful when the tmdb ID was known but the IMDb ID was missing or when the movie title in the script dataset matched the one in the tmdb dataset but not the IMDb dataset (some movie titles, especially foreign films, can change over time).

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/eda_scripts_out/num_movies_vs_year.png" width="60%">
</picture>
   
*Figure 3: Number of movies per year present in the script dataset.* 

#### Database updating
(see [data_updating.ipynb](https://github.com/nfasano/movie_recsys/blob/main/database/database_updating.ipynb) notebook for implementation)
After synthesizing and cleaning the datasets in the data wrangling phase, there still existed some missing data or errors that were discovered during the machine learning loop or after deployment. To fix these errors in an organized manner, I wrote a notebook (data_updating.ipynb) that allows one to easily update the cleaned database on a per-entry basis similar to how the SQLs UPDATE method works. 

____________________________________________________________________________________________

### Machine learning pipeline

#### Data preprocessing
(see [data_preprocessing.py](https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/data_preprocessing.py) notebook for implementation)

To process the script data for the LDA model, the following Natural Language Processing tasks were performed (see recsys_content_based/data_preprocessing_eda.ipynb notebook for implementation):
- stop word removal from NLTK list of stop words
- word mapping for words with the same meaning but different spellings (e.g. okay -> ok)
- lemmatization
- made the entire corpus lowercase
- removed punctuation from the entire corpus
- bag of words representation (create vector X of size num_movies by num_words)

The following figure shows the top 60 words and their word count across the entire corpus after the removal of stop words and words present in greater than 90% of the documents. For this figure, lemmatization was turned off during preprocessing which is why the words "friend" and "friends" appear. Here we see the diversity of words present in the corpus where the topic of family seems especially prevalent (e.g. words like mom, mother, baby, son, wife, father, and brother).

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/eda_scripts_out/word_count_top_100.png" alt="drawing" width="100%"/> 
</picture>
   
*Figure 4: Top 60 words contained in the entire corpus after removing stop words and words present in greater than 90% of the documents.* 

#### Model building
(see [model_building.ipynb](https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_building.ipynb) notebook for implementation)

The model used to find movies with similar content was Latent Dirichlet Allocation (LDA), which is a three-level hierarchical Bayesian model widely used for uncovering latent topics within large corpus [cite: Blei, et al. JMLR 2003](https://jmlr.csail.mit.edu/papers/v3/blei03a.html). In the context of this project, LDA is a generative probabilistic model that represents movie scripts (the documents) as a mixture of latent topics and represents each topic by a distribution of words. 

Importantly, LDA allows for documents to be represented by many different topics at the same time. Ideally, this will allow the model to distinguish between movies that have the same dominant topic but very different sub-topics. As an example, consider the movies 'Space Jam' and 'Remember the Titans.' Both films are predominantly about sports, but 'Space Jam' is also a comedy film geared toward younger audiences, and 'Remember the Titans' is also a biographical film about the end of segregation in American schools. (Note that LDA makes no guarantees about what the latent topics will represent or whether or not they will be interpretable as Genres. Nonetheless, I do find that the discovered topics are generally interpretable)

There are some drawbacks to the LDA model listed below.

Drawbacks to the LDA model:
1. The number of topics is fixed and known apriori
2. The topics are static and do not capture any time evolution
3. The model can produce topics with words that are uncorrelated, especially in noisy datasets

To determine the number of topics, we ran a 5-fold cross-validation in which 8000 were selected for the training dataset and 2000 samples were selected for the test set. Here we use the perplexity as the evaluation criteria. The perplexity is defined as exp(-1*log-likelihood per word) so the lower the perplexity, the better the model. The following figure shows the result of the cross-validation procedure, indicating that the reduction in perplexity with an increasing number of topics plateaus at ~20 topics (n_components ~ 20).

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_building_out/ncomp_tuning.png" alt="drawing" width="60%"/> 
</picture>
   
*Figure 5: Average perplexity evaluated on the training dataset (blue) and testing dataset (red) as a function of the number of components used to train the model. The average and standard deviation are computed after performing a 5-fold cross-validation.* 

#### Model evaluation
(see [model_eval.ipynb](https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval.ipynb) notebook for implementation)

Figure 5 shows the top words represented in the first 5 topics of the 20-topic model. Most of these topics are reasonably interpretable. Loosely speaking topic 1 represents war-type movies, topic 2 represents Christmas/holiday movies, topic 4 represents crime-type movies, and topic 5 represents sport-type movies. Topic 3 is a bit harder to label, but seems to represent a class of people, particularly when addressed formerly with words such as mr, mrs, dear, and darling. Each topic, however, does have some words that seem misplaced, such as harassment in topic 2 and the words nick, gwen, gandhi, jasper, and phil in topic 5.

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval_out/select_topics.png" alt="drawing" width="100%"/> 
</picture>
   
*Figure 6: Distribution of the top 20 words present in the first five topics of the 20-topic model.* 

The next figure shows a movie's distribution over the 20 topics and the top 15 words present in the two highest weighted topics for that movie for three different movies: (a) WALL-E, (b) Schindler's List, and (c) Remember the Titans.

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval_out/the_dark_knight_top_topics.png" alt="drawing" width="100%"/> 
</picture>
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval_out/a_christmas_carol,_2020_top_topics.png" alt="drawing" width="100%"/> 
</picture>
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval_out/remember_the_titans_top_topics.png" alt="drawing" width="100%"/> 
</picture>
   
*Figure 7: Three examples of a movie's distribution over the 20 topics and the top 20 words present in the two highest weighted topics for that movie. (a) WALL-E, (b) Schindler's List, and (c) Remember the Titans.* 

#### Ranking of recommendations by cosine-similarity and post-filtering options
Now that we have learned the distribution of topics for each movie and the distribution of words within each topic, we can compare the similarity between movie's topic distributions. To do that, we compute the cosine similarity between an input movie's topic distribution with the topic distribution of all other movies in the database and then rank the values in descending order. After the ranking by cosine-similarity is computed, the list of movies is filtered by removing all movies with an IMDb rating less than the rating_min input value (set rating_min to 0 for no filtering).

In figures 8 and 9 below, we show the recommendations provided for the input movies "Remember the Titans" and "Little Giants." Both films are about football teams but with very different contexts: Remember the Titans is while Little Giants is a comedy film about pee-wee football teams. Notice how the model captures these differences and recommends serious sports films for Remember the Titans and other family/comedy sports movies for Little Giants. Note that the model does not see the genre of the movie - only the script text.

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/readme_images/littlegiants_recommendations.png" alt="drawing" width="100%"/> 
</picture>
   
*Figure 8: Screenshots of the provided recommendations based on the input film "Little Giants."* 

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/readme_images/rememberthetitans_recommendations.png" alt="drawing" width="100%"/> 
</picture>
   
*Figure 9: Screenshots of the provided recommendations based on the input film "Remember the Titans."* 

____________________________________________________________________________________________

### Model deployment
(see [gradio_app.ipynb](https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/gradio_app.ipynb) notebook for implementation)
To deploy the developed model, we built a web-based app using [gradio](https://www.gradio.app/) and then hosted the app on [Hugging Face Spaces](https://huggingface.co/spaces), which allows the application to be available permanently and for free.

The code contains the following features:

1. The gradio code used to create the app. I chose to utilize gradio's low-level API ([Blocks](https://www.gradio.app/guides/quickstart#blocks-more-flexibility-and-control)) for building the app since it allows for more flexible layouts and data flows). 
2. A movie recommendation function (def movie_rec) that takes in a movie, some filters, and then returns the most similar movies (plus that movie's metadata) from the cosine-similarity matrix after filtering out low-ranking and Adult films. 
3. A Query parser (def update_radio(text)) that takes in the text from the search box and returns the most relevant movies from the dataset. The parser is case-insensitive and robust to punctuation and extraneous spaces, but it will not handle spelling mistakes. 

____________________________________________________________________________________________

### Model feedback and future work
After deploying the model and experimenting with several use cases, I learned several strengths and weaknesses of the current recommendation system implementation, which motivates future work. 

From the Ranking of recommendations section above, we see that the model does a great job of finding movies with similar themes and sub-themes. One drawback of the model becomes apparent when we ask for recommendations about movies that take place in the past century but are filmed in the current century. Take for example the following two recommendation lists for the movies "Green Book" and "The Help". For both movies, the recommender has done a reasonable job finding movies with similar content, but I am not sure that these movies are the most relevant recommendations to be made since all of the recommended movies were written and filmed before 1991 (for "The Help") and before 1980 (for "Green Book"). 

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/readme_images/thehelp_recommendations.png" alt="drawing" width="100%"/> 
</picture>
   
*Figure 10: Screenshots of the provided recommendations based on the input film "The Help."* 

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/readme_images/greenbook_recommendations.png" alt="drawing" width="100%"/> 
</picture>
   
*Figure 11: Screenshots of the provided recommendations based on the input film "Green Book."* 

One way to circumvent this problem would be to adjust the ranking algorithm by, for example, enforcing the criteria that 60% of recommendations must be of movies released in the last 10-20 years. Another approach would be to adjust the model itself. One extension to the LDA model, known as Dynamic LDA, attempts to model the evolution of topics over time to account for how the usage of certain words has evolved [see Blei, et al. ICML'06. (2006)](https://dl.acm.org/doi/abs/10.1145/1143844.1143859). A final way to improve the recommendations would be to tune the model parameters not for perplexity, but rather for some other downstream task, such as click-through rate, watch time, or some combination of metrics.

### Resources
[Microsoft Recommenders](https://github.com/recommenders-team/recommenders) - well-maintained GitHub repository detailing the best practices for building and deploying recommender systems.

[Nvida](https://docs.nvidia.com/deeplearning/performance/recsys-best-practices/index.html) - Document discussing best practices for building recommender systems

[Original LDA paper by Blei et al. JMLR 2003](https://jmlr.csail.mit.edu/papers/v3/blei03a.html)

[Dynamic LDA paper by Blei et al. ICML '06](https://dl.acm.org/doi/abs/10.1145/1143844.1143859)

[Example of collaborative Topic model used at NYT](https://archive.nytimes.com/open.blogs.nytimes.com/2015/08/11/building-the-next-new-york-times-recommendation-engine/?mcubz=0&_r=0) - Discusses the NYTs experimentation with collaborative topic models (LDA is used for topic model) in production.



