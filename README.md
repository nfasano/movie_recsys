# Content-based movie recommendation system

The deployed recommender can be tested by following [this link](https://nmfasano5-content-based-movie-recommendation-system.hf.space).

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/example_recsys_input.png" alt="drawing" width="800"/> 
</picture>

<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/recsys_content_based/images/example_recsys_output.png" alt="drawing" width="800"/> 
</picture>
</p>



## Introduction and Motivation
I often find myself saying the following phrase: "I wish I can go back in time and watch movie X for the first time again." The sentiment being that I know exactly what kind of content I want to consume, but at the same time, I want it to be new content. The goal of this project was to build a movie recommender that suggests new movies to watch that are similar in content to movie X. To do that, I built a topic model from 30,000+ film scripts using Latent Dirichlet Allocation (LDA). The film script data was webscraped from various sites and combined with additional metadata from publically available datasets. Movie recommendations are made based on the similarity between the latent topics of movie X with all other movies in the database.

## End-to-End Recommender System
The following figure shows the workflow used throughout this project. 

<p align="center">
<picture>
<img src="https://github.com/nfasano/movie_recsys/blob/main/movie_rec_pipeline.jpg" alt="drawing" width="800"/> 
</picture>
</p>

*Figure 1: Workflow chart used throughout this project.* 

### Data Engineering

#### Datasets used in this project:
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
  
### Machine Learning Loop
#### Data processing
#### Model building
#### Model evaluation

### Model deployment with Gradio and Hugging face spaces

### Deploying the recommender using gradio + Hugging Face's spaces




