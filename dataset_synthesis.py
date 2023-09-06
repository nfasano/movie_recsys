# Synthesize datasets from all sources into one dataframe that will be deployed with recommender
# This notebook should be run anytime the dataset is updated

import pandas as pd


def dataset_synthesis():
    """
    inputs:
            (none)
    outputs:
            df, a pd.DataFrame, synthesized scripts and IMDb dataset
    """

    # relative locations of csv snd tsv datasets
    path_to_scripts_csv = (
        "database_film_scripts\\springfield_movie_scripts_2023_01_13_clean.csv"
    )
    path_to_imdb_titles_tsv = "database_IMDb\\2023_02_12_IMDb_dataset\\titleBasics.tsv"
    path_to_imdb_ratings_tsv = (
        "database_IMDb\\2023_02_12_IMDb_dataset\\titleRatings.tsv"
    )

    # import (cleaned) movie script data and retain only 5 columns
    df = pd.read_csv(path_to_scripts_csv, index_col=[0])
    df = df[["movie_title", "movie_year", "imdb_id", "imdb_link", "tmdb_poster_link"]]

    # load in IMDb title basics and title ratings datasets
    df_imdb_basics = pd.read_csv(path_to_imdb_titles_tsv, sep="\t")
    df_imdb_basics = df_imdb_basics[["tconst", "genres", "isAdult"]]

    df_imdb_ratings = pd.read_csv(path_to_imdb_ratings_tsv, sep="\t")

    # join the two IMDb datasets on tconst (i.e. imdb_id)
    df_imdb = df_imdb_basics.join(
        other=df_imdb_ratings.set_index("tconst"), on="tconst", how="left"
    )

    # finally join scripts and IMDb datasets
    df = df.join(df_imdb.set_index("tconst"), on="imdb_id", how="left")

    # rename and reorder the columns
    df.columns = [
        "movie_title",
        "movie_year",
        "imdb_id",
        "imdb_link",
        "tmdb_poster_link",
        "genre",
        "is_adult",
        "average_rating",
        "num_votes",
    ]
    df = df[
        [
            "movie_title",
            "movie_year",
            "genre",
            "average_rating",
            "num_votes",
            "is_adult",
            "imdb_id",
            "imdb_link",
            "tmdb_poster_link",
        ]
    ]

    # add year to movie's with duplicate title names
    movie_title_counts = df["movie_title"].value_counts()
    movie_dups = movie_title_counts[movie_title_counts > 1]
    list_of_dups = movie_dups.index.tolist()

    df["movie_title"] = [
        jmovie_title + ", " + str(df["movie_year"].iloc[j])
        if jmovie_title in list_of_dups
        else jmovie_title
        for j, jmovie_title in enumerate(df["movie_title"])
    ]

    # clean up explicits in the movie titles
    df["movie_title"] = [
        jmovie.replace("uck", "***") if "fuck" in jmovie.lower() else jmovie
        for jmovie in df["movie_title"]
    ]
    df["movie_title"] = [
        jmovie.replace("itch", "****") if "bitch" in jmovie.lower() else jmovie
        for jmovie in df["movie_title"]
    ]
    df["movie_title"] = [
        jmovie.replace("hit", "***") if "shit " in jmovie.lower() else jmovie
        for jmovie in df["movie_title"]
    ]

    # save new dataframe for uploading to spaces
    return df
