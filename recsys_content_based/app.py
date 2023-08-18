import gradio as gr
import pickle
import requests
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv("df_spaces_upload.csv", index_col=[0])
df["tmdb_poster_link"] = df["tmdb_poster_link"].fillna("NULL")
df["imdb_link"] = df["imdb_link"].fillna("NULL")
# read in (precomputed) transformed movie matrix data (num_movies x num_topics)
with open("Xtran.txt", "rb") as f:
    Xtran = pickle.load(f)

# ----------------------------------------------------------------------------------- #
# ----------------------- variables used in functions ------------------------------- #
# ----------------------------------------------------------------------------------- #

# define some test cases
tested_examples = [
    ["Barbie", "5", True],
    ["Green Book", "6.9", True],
    ["Finding Nemo", "6", True],
    ["How to Train Your Dragon", "6.7", True],
    ["Remember the Titans", "7.1", True],
    ["Avengers: Endgame", "6.5", True],
]

# ----------------------------------------------------------------------------------- #
# --------------------------- function definitions ---------------------------------- #
# ----------------------------------------------------------------------------------- #


def construct_markdown_link(links, link_names):
    # Construct links in markdown style. Used in movie_rec function
    # when constructing the output dataframe
    return [
        "<span style='color: #0000EE;'> **<ins> ["
        + jname
        + "]("
        + jlink
        + ")</ins>**</span>"
        for jname, jlink in zip(link_names, links)
    ]


def set_output_visibility_true():
    # set visibility of recommendations dataframe to true when Recommend is clicked
    return [outputs[0].update(visible=True) for j in range(7)]


def set_output_visibility_false():
    # set visibility of recommendations dataframe to False when Reset is clicked
    return [outputs[0].update(visible=False) for j in range(7)]


def update_radio(text):
    # Update radio choices based on text in search_bar
    # input:
    #       text - string from search_bar
    # output:
    #       radio.update(choices=new_choices)

    if text is None:
        # make radio choices invisible
        return radio.update(choices=[], visible=False)

    # clean up input text of punctutation and extra white space
    text = re.sub(" +", " ", text.replace(":", " ").replace("'", "")).lower()

    if re.sub(" +", "", text) == "":
        # set radio to blank if only white space remains
        return radio.update(choices=[], visible=False)
    else:
        # find top three matches
        # Note: here we prefer hits where movie name starts with text

        # construct cleaned list of movie names
        movie_list = df["movie_title"].tolist()
        movie_list_clean = [j.lower() for j in movie_list]
        movie_list_clean = [
            re.sub(" +", " ", j.replace(":", " ").replace("'", ""))
            for j in movie_list_clean
        ]

        # find movie names that begin with text (order these at the top)
        top_hits_begin = [
            movie_list[j]
            for j, jname in enumerate(movie_list_clean)
            if jname.startswith(text)
        ]

        # find movie names that contain text (fill in as necessary)
        top_hits_in = [
            movie_list[j] for j, jname in enumerate(movie_list_clean) if text in jname
        ]

        # combine the two lists of movie names without duplication
        top_hits = top_hits_begin + [j for j in top_hits_in if j not in top_hits_begin]

        if len(top_hits) < 1:
            return radio.update(choices=[], visible=True, label="No movies found")
        else:
            top_hits = sorted(top_hits[:25])
            return radio.update(
                choices=top_hits[:3], visible=True, label="Select your movie:"
            )


def movie_rec(movie_name, rating_min, is_adult):
    # compute top 5 movie recommendations for the input movie and filters
    # inputs:
    #       movie_name: selected movie_name from radio
    #       rating_min: filter out all movies with ratings less than rating_min
    #       is_adult: if True then filter out adult titles
    # ouputs:
    #       df_in: dataframe with all the info on movie_name
    #       df_out: dataframe with all the info on top 5 recommended movies

    if not movie_name:
        raise gr.Error("Please select a movie before clicking Recommend")

    jmovie = df[df["movie_title"] == movie_name].index[0]
    sim_in = Xtran[jmovie, :].reshape(1, 20)

    if "NULL" in df["imdb_link"].iloc[jmovie]:
        # input movie has no matching IMDb title
        link_in = ["N/A"]
        genre_in = ["N/A"]
        rating_in = ["N/A"]
    else:
        link_in = construct_markdown_link([df["imdb_link"].iloc[jmovie]], [movie_name])
        genre_in = [df["genre"].iloc[jmovie]]
        rating_in = [df["average_rating"].iloc[jmovie]]

    # construct input dataframe
    df_in = pd.DataFrame(
        {
            "Title": [movie_name],
            "Year": [df["movie_year"].iloc[jmovie]],
            "IMDb Rating": rating_in,
            "Genres": genre_in,
            "IMDb Link": link_in,
        }
    )

    # compute similarity between movie_name and all other movies in database
    sim_movie = cosine_similarity(sim_in, Xtran).reshape((len(df),))

    # sort dataframe by movie similarity in descending order
    arg_sim_movie_ordered = np.flip(np.argsort(sim_movie))
    df_sort = df.iloc[arg_sim_movie_ordered[1:]]

    # fiter by rating_min and is_adult
    df_sort = df_sort[df_sort["average_rating"] >= float(rating_min)]
    if is_adult:
        df_sort = df_sort[df_sort["is_adult"] == 0]

    # raise error if less than 5 movies are left after filtering
    if len(df_sort) < 5:
        raise gr.Error(
            "Not enough movies met the filter criteria. Try reducing the minimum rating."
        )

    # construct output dataframe
    movie_title = df_sort["movie_title"].iloc[0:5].tolist()
    movie_year = df_sort["movie_year"].iloc[0:5].tolist()
    rating = df_sort["average_rating"].iloc[0:5].tolist()
    genre = df_sort["genre"].iloc[0:5].tolist()
    tmdb_poster_link = df_sort["tmdb_poster_link"].iloc[0:5].tolist()
    link = construct_markdown_link(df_sort["imdb_link"].iloc[0:5].tolist(), movie_title)

    df_out = pd.DataFrame(
        {
            "Title": movie_title,
            "Year": movie_year,
            "IMDb Rating": rating,
            "Genres": genre,
            "IMDb Link": link,
        }
    )

    movie_poster_link = []
    for j in range(5):
        if tmdb_poster_link[j] == "NULL":
            movie_poster_link.append(None)
        else:
            movie_poster_link.append(tmdb_poster_link[j])

    return df_in, df_out, *movie_poster_link


# ----------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------- #
# ----------------------- begin interface construction ------------------------------ #
# ----------------------------------------------------------------------------------- #
# ----------------------------------------------------------------------------------- #
theme = gr.themes.Default().set(
    body_text_color="*neutral_950",
    body_text_color_subdued="*body_text_color",
    link_text_color="*body_text_color",
    link_text_color_active="*body_text_color",
    link_text_color_hover="*body_text_color",
    block_info_text_color="*neutral_950",
    block_label_text_color="*neutral_950",
    block_title_text_color="*neutral_950",
    block_background_fill="*neutral_100",
)


with gr.Blocks(theme=theme) as demo:
    # ----------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------- #
    # --------------------------------- rec_sys tab ------------------------------------- #
    # ----------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------- #
    with gr.Tab("Recommendation System"):
        gr.HTML("<h2 align='center'> Content-Based Movie Recommendation System </h2>")
        gr.Markdown(
            """ Welcome! This app helps you find great films with content that is closely related to your favorite movie.
                To get started, search and select a movie from our database. Then set your filters and click Recommend to get a list of movie recommendations.
                Just want to get a feel for how the app works? Try selecting one of the preconfigured examples below and click Recommend. 
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                search_bar = gr.Text(
                    label="Search box", placeholder="Start typing here"
                )
                radio = gr.Radio(choices=[], label="Select a movie", visible=False)
                search_bar.change(update_radio, inputs=search_bar, outputs=radio)
                radio.change(search_bar.update, inputs=radio, outputs=search_bar)

                gr.Markdown("Recommendation filters")
                with gr.Row():
                    rating_in = gr.Slider(
                        0,
                        10,
                        value=0,
                        step=0.1,
                        label="Enter minimum rating",
                        scale=1,
                    )
                    is_adult = gr.Checkbox(
                        value=True,
                        info="Exclude adult films?",
                        label="Non-adult films only",
                        scale=1,
                    )
            with gr.Column(scale=1):
                gr.Examples(
                    tested_examples,
                    [radio, rating_in, is_adult],
                    label="Examples: Select a preconfigured option below and click Recommend",
                )
                with gr.Row():
                    rec_btn = gr.Button("Recommend", variant="primary")
                    reset_btn = gr.ClearButton(
                        [search_bar, radio, rating_in], value="Reset"
                    )
        with gr.Row():
            with gr.Column(scale=36):
                outputs = [
                    gr.DataFrame(
                        label="Input Movie",
                        headers=[
                            "Title",
                            "Year",
                            "IMDb Rating",
                            "Genres",
                            "IMDb Link",
                        ],
                        row_count=(1, "fixed"),
                        col_count=(5, "fixed"),
                        interactive=False,
                        datatype=["str", "str", "str", "str", "markdown"],
                        visible=False,
                    ),
                    gr.DataFrame(
                        label="Your Recommendations",
                        headers=[
                            "Title",
                            "Year",
                            "IMDb Rating",
                            "Genres",
                            "IMDb Link",
                        ],
                        row_count=(5, "fixed"),
                        col_count=(5, "fixed"),
                        interactive=False,
                        datatype=["str", "str", "str", "str", "markdown"],
                        visible=False,
                    ),
                ]
                with gr.Row():
                    for j in range(5):
                        outputs.append(
                            gr.Image(
                                type="filepath",
                                show_download_button=False,
                                visible=False,
                                show_label=False,
                                container=False,
                            )
                        )

                outputs[0].change(set_output_visibility_true, outputs=outputs)

        rec_btn.click(
            fn=movie_rec,
            inputs=[radio, rating_in, is_adult],
            outputs=outputs,
            scroll_to_output=True,
        )

        reset_btn.click(set_output_visibility_false, outputs=outputs)

    # ----------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------- #
    # --------------------------------- Details Tab ------------------------------------- #
    # ----------------------------------------------------------------------------------- #
    # ----------------------------------------------------------------------------------- #

    with gr.Tab("Details"):
        gr.HTML("<h1 align='center'> Content-Based Movie Recommendation System </h1>")
        gr.HTML("<h3 align='center'> By Nicholas Fasano </h3>")
        gr.Markdown(
            """
            <p style='text-align: center;'> 
            Find me on <a href='https://github.com/nfasano/'>Github</a> 
            and <a href='https://www.linkedin.com/in/nmfasano/'>linkedIn</a> 
            </p>
            """
        )
        gr.Markdown("Insert details about recommender system here with some plots")

demo.launch()
