# Content-based movie recommendation system

## Introduction: motivation, dataset and model overview, and the deployed recommender 
The goal of this project was to build a movie recommender that suggests new movies to watch that are similar in content to an input movie. It helps users find potentially unknown movies that have similar topics to their favorite movies. To accomplish this, I set out to  Latent Dirichlet allocations (LDA) was used to find the latent topics of the entire corpus and cosine-similarity was used to find pairs of movies with similar latent topics. The film script dataset was webscraped from various sites and combined with additional metadata from publically available datasets (IMDb.com, tmdb.org, and MovieLens.com). 

The deployed recommender can be tested by following [this link](https://nmfasano5-content-based-movie-recommendation-system.hf.space). A screenshot showing some example recommendations based on the movie input "Remember the Titans" is shown in the following figure. The remainder of this README.md file details the workflow used to build the recommender system, including the data engineering and machine learning loops which allow for seamless improvements to any part of the system.

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/example_recsys_input.png" alt="drawing" width="850"/> 
</picture>

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/example_recsys_output.png" alt="drawing" width="850"/> 
</picture>
</p>

*Figure 1: Snapshot of the movie recommender web app deployed to Hugging Face's spaces. In this example, the user began searching for a movie and then selected "Remember the Titans" from the list of available movies. The user chooses various filters to be applied to the recommendations and clicks Recommend. At this point, 5 movie titles along with some metadata and IMDb.com links to the movies' title pages are shown.* 

## The Life-Cycle of this Movie Recommender Project
The following figure shows the life-cycle of this movie recommendation project, highlighting the data engineering and machine learning loops used to continuously make improvements to the recommender system. 

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_rec_pipeline.jpg" alt="drawing" width="800"/> 
</picture>
</p>

*Figure 2: Life-cycle diagram for this recommender system.* 

The diagram begins with data engineering, where we collected disparate datasets through webscrapping of movie script data and . These datasets are then cleaned and synthesized . Functions are written to continuously clean and update the database for improving the quality of data for training the machine learning model.

With a cleaned and synthesized dataset, the machine learning loop can begin. In this phase of the project, we first began by doing some exploratory data analysis to understand the richness of the dataset and then preprocessed the datasets (created a bag of words representation) into a form ready to be used for training of the model.


### Data Engineering

#### Curated datasets used in this project:
1) Movie and television scripts from IMSDb and Springfield! Springfield!
    - Source: https://imsdb.com/ and https://www.springfieldspringfield.co.uk/
    - Content: film title, year, film script for 35,000+ movies and 130,000+ tv epsiodes 
    - Note: This dataset was webscraped. See [scrapers](https://github.com/nfasano/movie_recsys/tree/main/database_film_scripts) folder in this repo for the notebooks used to scrape the scripts.
2) IMDb dataset
    - Source: [IMDb dataset](https://www.imdb.com/interfaces/)
    - Content: film title, average rating, genres, runtime, cast/crew 
    - Note: This dataset is freely available for noncommercial use.
3) The Movie DataBase (TMBD)
    - Source: [themoviedb.org](https://www.themoviedb.org/?language=en-US)
    - Note: This dataset is freely available for noncommercial use. It was primarily used for adding poster art to the recommender app and for matching ids between scripts dataset and imdb dataset.
4) MovieLens Dataset
    - Source: [MovieLens.com](https://movielens.org/home)
    - Content: user_id, item_id, rating, timestamp for over 20 Million ratings provided by real users on the MovieLens website. Also provides the ids for IMDb and TMDB for all movies in their database, making it trivial to combine this dataset with IMDb and TMDB metadata.
    - Note: This dataset is freely available for noncommercial use.

#### Data wrangling
(see recsys_content_based/data_cleaning_and_synthesis.ipynb notebook for implementation)
To make this data usable to train our machine learning models and be available as metadata for the recommender app, it is necessary to clean and synthesize the datasets into one database. To clean the webscraped data, we manually inspected the movie titles and year of release and fixed any inconsistencies or errors, and changed the movie title formatting to be the same as the IMDb and tmdb datasets. Some examples of this data cleaning are shown below:

1) Fixed errors -- e.g. changed movie year from "0000" to "1999" or from "20147" to "2017"
2) Put titles into consistent formatting across datasets -- e.g. changed movie_titles in script dataset from "Dark Knight, The" to "The Dark Knight" or from "Beautiful Mind, A" to "A Beautiful Mind".
3) Removed duplicate entries with identical (or nearly identical) scripts
4) Removed data entries that only contained irrelevant script_text such as "None" or "More Movie Scripts | Request a Movie Transcript"

The next step was to find matching title IDs from the websracpped script dataset with he IMDb and tmdb datasets. A few different approaches were taken to systematically find these IDs, such as matching movie titles and movie year between the script dataset and IMDb/tmdb datasets. Additionally, the tmdb.org website provides a convenient API that can be queried for all data using either tmdb ID, IMDb ID, or movie title and year, which allows for finding all relevant movie data when given any one of these three. This API was particularly helpful when the tmdb ID was known but the IMDb ID was missing or when the movie title in the script dataset matched the one in the tmdb dataset but not the IMDb dataset (some movies titles, especially foreign films, can change over time).

#### Database updating
(see recsys_content_based/data_updating.ipynb notebook for implementation)
After synthesizing and cleaning the datasets in the data wrangling phase, there still existed some missing data or errors that were discovered during the machine learning loop or after deployment. To fix these errors in an organized manner, I wrote a notebook (data_updating.ipynb) that allows one to easily update the cleaned database on a per-entry basis similar to how SQLs UPDATE method works. 
  
### Machine Learning Loop
#### Data preprocessing
To process the script data for the LDA model, the following Natural Language Processing tasks were performed (see recsys_content_based/data_preprocessing_eda.ipynb notebook for implementation):
- stop word removal from NLTK list of stop words
- word mapping for words with the same meaning but different spellings (e.g. okay -> ok)
- lemmatization
- made the entire corpus lowercase
- removed punctuation from entire corpus
- bag of words representation (create vector X of size num_movies by num_words)

<!---
<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/data_preprocessing_eda_out/num_movies_vs_year.png" alt="drawing" width="800"/> 
</picture>
</p>
--->

The following figure shows the top 60 words and their word count across the entire corpus after the removal of stop words and words present in greater than 90% of the documents. For this figure, lemmatization was turned off during preprocessing which is why the words "friend" and "friends" appear. Here we see the diversity of words present in the corpus where the topic of family seems especially prevalent (e.g. words like mom, mother, baby, son, wife, father, brother).

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/data_preprocessing_eda_out/word_count_top_100.png" alt="drawing" width="800"/> 
</picture>
</p>

#### Model building
The model used to find movies with similar content was Latent Dirichlet Allocation (LDA), which is a three-level hierarchical Bayesian model widely used for uncovering latent topics within large corpus [cite: Blei, et al. Jorunal of Machine Learning Research 2003]. In the context of this project, LDA is a generative probabilistic model that represents movie scripts (the documents) as a mixture of latent topics and represents each topic by a distribution of words. 

Importantly, LDA allows for documents to be represented by many different topics at the same time. Ideally, this will allow the model to distinguish between movies that have the same dominant topic but very different sub-topics. As an example, consider the movies 'Space Jam' and 'Remember the Titans.' Both films are predominantly about sports, but 'Space Jam' is also a comedy film geared toward younger audiences, and 'Remember the Titans' is also a biographical film about the end of segregation in American schools. (Note that LDA makes no guarantees about what the latent topics will represent or whether or not they will be interpretable as Genres. Nonetheless, I do find that the discovered topics are generally interpretable)

There are some drawbacks to the LDA model listed below. Follow-up works to the LDA model, including by Blei himself, are available. In the next section we discuss one of these model variants (dynamic LDA) that may be particularly useful for this dataset and motivates future work.

Drawbacks to the LDA model:
1. the number of topics is fixed and known apriori
2. the topics are static and do not capture any time evolution
3. can produce topics with words that are uncorrelated, especially in noisy datasets

To determine the number of topics, n_components in the code base, we train a range of LDA models with different numbers of topics and evaluate the perplexity on a held-out test set of data. The perplexity is defined as exp(-1*log-likelihood per word) so the lower the perplexity, the better the model. The following figure shows the result of this hyperparameter scan, indicating that the reduction in perplexity with an increasing number of topics plateaus at ~20 topics.

#### Model evaluation

#### High-level features of the model
The following figure shows the top words represented in the first 5 topics of the 20-topic model. Most of these topics are readily interpretable as mentioned previously. Loosely speaking topic 1 represents war-type movies, topic 2 represents Christmas/holiday movies, topic 4 represents crime-type movies, and topic 5 represents sport type movies. Topic 3 is a bit harder to label, but seems to represent a class of people, particularly when addressed formerly with words such as mr, mrs, dear, darling. Each topic, however, does have some words that seem misplaced, such as harassment in topic 2 and the words nick, gwen, gandhi, jasper, and phil in topic 5.

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval/select_topics.png" alt="drawing" width="800"/> 
</picture>
</p>

The next figure shows a movie's distribution over the 20 topics and the top 20 words present in the top two topics for that movie for three different movies: (a) WALL-E, (b) Schindler's List, and (c) Remember the Titans.

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval/walle_top_topics.png" alt="drawing" width="800"/> 
</picture>
    
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval/schindlers_list_top_topics.png" alt="drawing" width="800"/> 
</picture>

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_eval/remember_the_titans_top_topics.png" alt="drawing" width="800"/> 
</picture>
</p>

#### Ranking of recommendations by cosine-similarity and post-filtering options
Now that we have learned the distribution of topics for each movie and the distribution of words within each topic, we can now compare the similarity between movies based on how similar their topic distributions are. To do that, we compute the cosine-similarity between an input movie's topic distribution with the topic distribution of all other movies in the database and the rank the values in descending order.  

After the ranking by cosine-similarity is computed, the list of movies is filtered by removing all movies with an IMDb rating less than the rating_min input value (set rating_min to 0 for no filtering).

The following figure shows the output recommendations for these test examples. As you can see, the model does an excellent job of finding movies with similar underlying content. For example, take a look at the recommendations provided for the movies "Remember the Titans" and "Little Giants". Both films are about football teams but with very different contexts: Remember the Titans is while Little Giants is a comedy film about pee-wee football teams. Notice how the recommendations captures these differences and recommends serious sports films for Remember the Titans and other family/comedy sports movies for Little Giants. Note that the model does not see the genre of the movie - only the script text.

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/littlegiants_recommendations.png" alt="drawing" width="750"/> 
    
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/rememberthetitans_recommendations.png" alt="drawing" width="750"/> 
</picture>
</p>

### Model deployment with Gradio and Hugging Face's spaces
To deploy the developed model, we built a web-based app using [gradio](https://www.gradio.app/) and then hosted the app on [Hugging Face Spaces](https://huggingface.co/spaces), which allows the application to be available permanently and for free.

The code contains the following features:

1. The gradio code used to create the app. I chose to utilize gradio's low-level API ([Blocks](https://www.gradio.app/guides/quickstart#blocks-more-flexibility-and-control)) for building the app, since it allows for more flexible layouts and data flows). 
2. A movie recommendation function (def movie_rec) that takes in a movie, some filters, and then returns the most similar movies (plus that movies metadata) from the cosine-similarity matrix after filtering out low-ranking and Adult films. 
3. A Query parser (def update_radio(text)) which takes in the text from the search box and returns the most relevant movies from the dataset. The parser is case-insensitive and robust to punctuation and extraneous spaces, but it will not handle spelling mistakes. 


### Model feedback -- future work
After deploying the model and experimenting with several use cases, I learned several strengths and weaknesses of the current recommendation system implementation, which motivates future work. 

From the Ranking of recommendations section above, we see that the model does a great job of finding movies with similar themes and sub-themes. One drawback of the model becomes apparent when we ask for recommendations about movies that take place in the past but written/produced with modern technology. Take for example the following two recommendation lists for the movies "Green Book" and "The Help". For both movies, the recommender has done a reasonable job finding movies with similar content, but I am not sure that these movies are the most relevant recommendations to be made since all of the recommended movies were written and filmed prior to 1991 (for "The Help") and prior to 1980 (for "Green Book"). 

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/thehelp_recommendations.png" alt="drawing" width="750"/> 
    
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/greenbook_recommendations.png" alt="drawing" width="750"/> 
</picture>
</p>

One way to circumvent this problem would be to adjust the ranking algorithm by, for example, enforcing the criteria that 60% of recommendations must be of movies released in the last 10 years. Another approach would be to adjust the model itself. One extension to the LDA model, known as Dynamic LDA, attempts to model the evolution of topics over time to account for the way in which the usage of certain words has evolved [see Blei, et al. ICML'06. (2006)]. A final way to improve the recommendations would be to tune the model parameters not for perplexity, but rather for some other downstream task, such as click-through rate.




