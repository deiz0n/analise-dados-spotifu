import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

st.sidebar.title("Upload dos Arquivos:")


def upload_datasets():
    spotify_and_youtube = st.sidebar.file_uploader("Carregar dados do Spotify e YouTube", type=["csv"])
    top_in_spotify = st.sidebar.file_uploader("Carregar dados do Spotify", type=["csv"])

    if spotify_and_youtube is not None and top_in_spotify is not None:
        # Ler os arquivos para DataFrames
        spotify_youtube_df = pd.read_csv(spotify_and_youtube)
        top_spotify_df = pd.read_csv(top_in_spotify)
        return spotify_youtube_df, top_spotify_df
    else:
        return None, None


spotify_youtube, top_spotify = upload_datasets()

st.title("Análise de Música: Spotify e YouTube")
st.sidebar.title("Navegação")
page = st.sidebar.radio("Selecione a página:", ["Home", "Others"])


def __clean_artist_name(name):
    return ''.join(e for e in name if e.isalnum() or e.isspace())


def __clean_song_name(name):
    return ''.join(e for e in name if e.isalnum() or e.isspace())


def __filtro_global():
    artistas_unicos = spotify_youtube['Artist'].apply(__clean_artist_name).unique()
    artista_selecionado = st.sidebar.selectbox("Selecione um Artista:", ["Todos"] + list(artistas_unicos))
    if artista_selecionado != "Todos":
        return spotify_youtube[spotify_youtube['Artist'] == artista_selecionado]
    return spotify_youtube


def __format_y_axis(value, pos):
    return '{:,.0f}'.format(value)


spotify_youtube_filtrado = __filtro_global()


def numbers_streams_and_views():
    spotify_youtube_filtrado['Artist'] = spotify_youtube_filtrado['Artist'].apply(__clean_artist_name)
    st.subheader("Artistas com Mais Streams vs. Views no YouTube")

    streams_views = spotify_youtube_filtrado.groupby("Artist")[["Stream", "Views"]].sum().reset_index()
    top_20_streams = streams_views.nlargest(20, 'Stream')
    top_20_melted = top_20_streams.melt(id_vars="Artist", var_name="Platform", value_name="Count")

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=top_20_melted, x="Artist", y="Count", hue="Platform", marker='o', palette="viridis")
    plt.xticks(rotation=90)

    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(__format_y_axis))

    plt.xlabel("Artista")
    plt.ylabel("Contagem")
    st.pyplot(plt)


def danceability_vs_views():
    st.subheader("Relação entre Danceability e Views")
    plt.figure(figsize=(10, 6))

    sns.scatterplot(
        data=spotify_youtube_filtrado,
        x="Danceability",
        y="Views",
        hue="Energy",
        palette="viridis",
        alpha=0.7
    )

    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(__format_y_axis))

    plt.xlabel("Danceability")
    plt.ylabel("Views")
    st.pyplot(plt)


def correlation_attributes():
    st.subheader("Heatmap de Correlação entre Atributos")
    numeric_data = spotify_youtube_filtrado.select_dtypes(include='number')
    correlation = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="viridis", square=True)
    st.pyplot(plt)


def average_like_by_artist():
    spotify_youtube_filtrado['Artist'] = spotify_youtube_filtrado['Artist'].apply(__clean_artist_name)

    likes_by_artist = spotify_youtube_filtrado.groupby("Artist")["Likes"].mean().reset_index()

    likes_by_artist = likes_by_artist.sort_values(by="Likes", ascending=False).head(10)

    st.subheader("Média de Likes por Artista")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=likes_by_artist, x="Artist", y="Likes", palette="viridis")
    plt.xticks(rotation=90)

    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(__format_y_axis))

    plt.xlabel("Artista")
    plt.ylabel("Média de Likes")
    st.pyplot(plt)


def top_artists_pie_charts():
    spotify_youtube_filtrado['Artist'] = spotify_youtube_filtrado['Artist'].apply(__clean_artist_name)
    st.subheader("Artistas com Mais Streams no Spotify e Views no YouTube")

    streams_views = spotify_youtube_filtrado.groupby("Artist")[["Stream", "Views"]].sum().reset_index()
    top_10_spotify = streams_views.nlargest(10, 'Stream')
    top_10_youtube = streams_views.nlargest(10, 'Views')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.pie(top_10_spotify['Stream'], labels=top_10_spotify['Artist'], autopct='%1.1f%%', startangle=140)
    ax1.set_title("Artistas como mais Streams no Spotify")
    ax2.pie(top_10_youtube['Views'], labels=top_10_youtube['Artist'], autopct='%1.1f%%', startangle=140)
    ax2.set_title("Artistas como mais Views no YouTube")

    plt.tight_layout()
    st.pyplot(fig)


def top_songs_metrics():
    spotify_youtube_filtrado['Track'] = spotify_youtube_filtrado['Track'].apply(__clean_song_name)
    metrics_per_song = spotify_youtube_filtrado.groupby("Track")[
        ["Stream", "Views", "Likes", "Comments"]].sum().reset_index()
    top_10_songs = metrics_per_song.nlargest(10, 'Stream')
    top_10_melted = top_10_songs.melt(id_vars="Track", var_name="Metric", value_name="Count")

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_10_melted, x="Track", y="Count", hue="Metric", palette="viridis")
    plt.xticks(rotation=45, ha="right")

    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(mticker.FuncFormatter(__format_y_axis))
    st.subheader("Músicas com maior número de streams, views, likes e comentários")
    plt.xlabel("Música")
    plt.ylabel("Contagem")
    st.pyplot(plt)


def valence_acousticness_loudness():
    st.subheader("Relação entre Valence, Acousticness e Loudness")
    spotify_youtube_filtrado["Valence"] = spotify_youtube_filtrado["Valence"].astype(str).str.replace(',', '.').astype(
        float)
    spotify_youtube_filtrado["Acousticness"] = spotify_youtube_filtrado["Acousticness"].astype(str).str.replace(',',
                                                                                                                '.').astype(
        float)
    spotify_youtube_filtrado["Loudness"] = spotify_youtube_filtrado["Loudness"].astype(str).str.replace(',',
                                                                                                        '.').astype(
        float)
    data = spotify_youtube_filtrado[["Valence", "Acousticness", "Loudness"]].dropna()

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        x=data["Valence"],
        y=data["Acousticness"],
        s=data["Loudness"] * 10,
        c=data["Loudness"],
        cmap="viridis",
        alpha=0.6,
        edgecolor="w"
    )

    plt.colorbar(scatter, label="Loudness")
    plt.xlabel("Valence")
    plt.ylabel("Acousticness")
    st.pyplot(plt)


def tabela_expandida_popularidade():
    st.subheader("Tabela Expandida de Atributos com Classificação de Popularidade por Artista")

    spotify_youtube_filtrado["Danceability"] = spotify_youtube_filtrado["Danceability"].astype(str).str.replace(',',
                                                                                                                '.').astype(
        float)
    spotify_youtube_filtrado["Energy"] = spotify_youtube_filtrado["Energy"].astype(str).str.replace(',', '.').astype(
        float)
    spotify_youtube_filtrado["Tempo"] = spotify_youtube_filtrado["Tempo"].astype(str).str.replace(',', '.').astype(
        float)
    spotify_youtube_filtrado["Speechiness"] = spotify_youtube_filtrado["Speechiness"].astype(str).str.replace(',',
                                                                                                              '.').astype(
        float)

    mean_attributes = spotify_youtube_filtrado.groupby("Artist")[
        ["Danceability", "Energy", "Tempo", "Speechiness"]].mean().reset_index()
    mean_attributes["Popularity Score"] = (mean_attributes["Danceability"] * 0.3 +
                                           mean_attributes["Energy"] * 0.4 +
                                           mean_attributes["Tempo"] * 0.2 +
                                           mean_attributes["Speechiness"] * 0.1)

    mean_attributes = mean_attributes.sort_values(by="Popularity Score", ascending=False).reset_index(drop=True)
    st.dataframe(
        mean_attributes.style
        .highlight_max(axis=0, color='lightgreen')
        .set_caption("Médias e Score de Popularidade por Artista")
    )


if page == "Home":
    numbers_streams_and_views()
    danceability_vs_views()
    correlation_attributes()
    average_like_by_artist()
elif page == "Others":
    top_artists_pie_charts()
    top_songs_metrics()
    valence_acousticness_loudness()
    tabela_expandida_popularidade()
