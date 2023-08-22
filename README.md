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
    - Note: This dataset was webscraped. See [scrapers](https://github.com/nfasano/movie_recsys/tree/main/database_film_scripts) folder in this repo for the notebooks used to scrape the scripts and clean the datasets.
2) IMDb dataset
    - Source: [IMDb dataset](https://www.imdb.com/interfaces/)
    - Content: film title, average rating, genres, runtime, cast/crew 
    - Note: This dataset is freely available for noncommercial use.
3) The Movie DataBase (TMBD)
    - Source: [themoviedb.org](https://www.themoviedb.org/?language=en-US)
    - Note: This dataset is freely available for noncommercial use. It was primarily used for adding poster art to the recommender app.
4) MovieLens Dataset
    - Source: [MovieLens.com](https://movielens.org/home)
    - Content: user_id, item_id, rating, timestamp for over 20 Million ratings provided by real users on the MovieLens website. Also provides the ids for IMDb and TMDB for all movies in their database, making it trivial to combine this dataset with IMDb and TMDB metadata.
    - Note: This dataset is freely available for noncommercial use.

#### Data wrangling
To make this data usable to train our machine learning models and be available as metadata for the recommender app, it is necessary to clean and synthesize the datasets into one database. To clean the webscraped data, we manually inspected the movie titles and year of release and fixed any inconsistencies or errors, and changed the movie title formatting to be the same as the IMDb and tmdb datasets. Some examples of this data cleaning are shown below:

1) Fixed errors -- e.g. changed movie year from "0000" to "1999" or from "20147" to "2017"
2) Put titles into consistent formatting across datasets -- e.g. changed movie_titles in script dataset from "Dark Knight, The" to "The Dark Knight" or from "Beautiful Mind, A" to "A Beautiful Mind".
3) Removed duplicate entries with identical (or nearly identical) scripts
4) Removed data entries that only contained irrelevant script_text such as "None" or "More Movie Scripts | Request a Movie Transcript"

The next step was to find matching title IDs from the websracpped script dataset with he IMDb and tmdb datasets. A few different approaches were taken to systematically find these IDs, such as matching movie titles and movie year between the script dataset and IMDb/tmdb datasets. Additionally, the tmdb.org website provides a convenient API that can be queried for all data using either tmdb ID, IMDb ID, or movie title and year, which allows for finding all relevant movie data when given any one of these three. This API was particularly helpful when the tmdb ID was known but the IMDb ID was missing or when the movie title in the script dataset matched the one in the tmdb dataset but not the IMDb dataset (some movies titles, especially foreign films, can change over time).

#### Database updating
After synthesizing and cleaning the datasets in the data wrangling phase, there still existed some missing data or errors that were discovered during the machine learning loop or after deployment. To fix these errors in an organized manner, I wrote a notebook (data_updating.ipynb) that allows one to easily update the cleaned database on a per-entry basis similar to how SQLs UPDATE method works. 
  
### Machine Learning Loop
#### Data preprocessing
To process the script data for the LDA model, the following Natural Language Processing tasks were performed:
- stop word removal from NLTK list of stop words
- word mapping for words with the same meaning but different spellings (e.g. okay -> ok)
- lemmatization
- made the entire corpus lowercase
- removed punctuation from entire corpus
- bag of words representation (create vector X of size num_movies by num_words)


<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/data_preprocessing_eda_out/num_movies_vs_year.png" alt="drawing" width="800"/> 
</picture>
</p>

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/data_preprocessing_eda_out/word_count_top_100.png" alt="drawing" width="800"/> 
</picture>
</p>

#### Model building
- insert LDA topics plot and some topic distributions for certain movies

Some drawbacks to the LDA model:
- the number of topics is fixed and known apriori
- the topics are static and do not capture any time evolution
- can produce topics with words that are uncorrelated, especially in noisy datasets

Follow-up works to the LDA model, including by Blei himself, are available. In the next section we discuss one of these model variants (dynamic LDA) that may be particularly useful for this dataset and motivates future work.

#### Model evaluation
- compare two models in a heuristic fashion and show the results
<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_building_and_eval/remember_the_titans_top_topics.png" alt="drawing" width="800"/> 
</picture>
</p>

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_building_and_eval/schindlers_list_top_topics.png" alt="drawing" width="800"/> 
</picture>
</p>

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_building_and_eval/walle_top_topics.png" alt="drawing" width="800"/> 
</picture>
</p>

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/model_building_and_eval/select_topics.png" alt="drawing" width="800"/> 
</picture>
</p>

### Model deployment with Gradio and Hugging Face's spaces
Explain gradio code in a few words, include some links




