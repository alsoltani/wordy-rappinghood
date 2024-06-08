import ast
import logging
import re
from pathlib import Path
from typing import List

import duckdb
import numpy.typing as npt
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNK_SIZE = 10_000

logging.basicConfig(level=logging.WARNING)


class DuckDBClient:
    DATABASE_NAME = "embeddings.duckdb"

    def __init__(
        self,
        embedding_size: int = 384,
        logger: logging.Logger = logging.getLogger(__name__),
    ):
        self.connection = duckdb.connect(self.DATABASE_NAME)
        self.embedding_size = embedding_size
        self.logger = logger
        self._columns: List[str] = []

    @property
    def columns(self):
        """
        Get correct column order from table.
        """
        if self._columns is None:
            self.connection.execute("PRAGMA table_info('tracks');")
            table_info = self.connection.fetchall()
            self._columns = [i[1] for i in table_info]
        return self._columns

    def migrate(self):
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS tracks (
                track_id INTEGER PRIMARY KEY,
                album_engineer VARCHAR,
                album_information VARCHAR,
                album_producer VARCHAR,
                album_tags VARCHAR,
                album_title VARCHAR,
                album_type VARCHAR,
                artist_associated_labels VARCHAR,
                artist_bio VARCHAR,
                artist_members VARCHAR,
                artist_name VARCHAR,
                artist_tags VARCHAR,
                track_composer VARCHAR,
                track_genre_top VARCHAR,
                track_genres VARCHAR,
                track_information VARCHAR,
                track_lyricist VARCHAR,
                track_publisher VARCHAR,
                track_tags VARCHAR,
                track_title VARCHAR
            );
            """
        )
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS embeddings (
                track_id INTEGER PRIMARY KEY,
                vector FLOAT[{self.embedding_size}],
            );
            """
        )
        self.connection.execute(
            """
            -- Manually load vector similarity search.
            -- Allow persistence to create index outside of in-memory db.
            INSTALL vss;
            LOAD vss;
            SET hnsw_enable_experimental_persistence = TRUE;
            """
        )
        self.connection.execute(
            f"""
            /* 
            Create an HNSW cosine index
            to accelerate array distance queries.
            */
            CREATE INDEX IF NOT EXISTS cos_index ON embeddings USING HNSW (vector) 
            WITH (metric = 'cosine');
            """
        )

    def insert_tracks(self, tracks: pd.DataFrame) -> None:
        """
        Improve UPSERT on DuckDB by loading from a temp JSON file,
        to limit row-by-row overhead.
        """
        temp_filepath = "temp_tracks.json"
        assert tracks.index.name == "track_id"

        # Flatten multi-index frame
        tracks.columns = pd.Index(["_".join(c) for c in tracks.columns.to_flat_index()])

        # Encore correct column order
        tracks.reset_index()[self.columns].to_json(temp_filepath, orient="records")

        self.connection.execute(
            f"""
            INSERT OR REPLACE INTO tracks
                SELECT * FROM read_json_auto("{temp_filepath}")
            """
        )

        self.logger.info(
            f"Successfully inserted track entries: {tracks.index.tolist()}.",
            extra={"table": "tracks"},
        )

        # Delete temp file
        # temp_file = Path(temp_filepath)
        # temp_file.unlink()

    def insert_embeddings(self, embeddings: pd.DataFrame) -> None:
        temp_filepath = "temp_embeddings.json"
        embeddings.to_json(temp_filepath, orient="records")

        self.connection.execute(
            f"""
            INSERT OR REPLACE INTO embeddings
                SELECT * FROM read_json_auto("{temp_filepath}")
            """
        )

        self.logger.info(
            f"Successfully inserted track entries: {embeddings['track_id'].tolist()}.",
            extra={"table": "embeddings"},
        )

        # Delete temp file
        temp_file = Path(temp_filepath)
        temp_file.unlink()

    def get_nearest_tracks_info(self, embedding: npt.NDArray, n_results: int = 3):
        print(type(embedding))
        self.connection.execute(
            f"""
        SELECT track_id, artist_name, track_title, track_genres
        FROM embeddings
        LEFT JOIN tracks
        USING (track_id)
        ORDER BY array_distance(vector, {embedding.tolist()}::FLOAT[{self.embedding_size}])
        LIMIT {n_results};
        """
        )
        return self.connection.fetchall()

    def close(self):
        self.connection.close()


class DataProcessor:
    TAGS_GENRES_COLUMNS = [
        ("track", "tags"),
        ("album", "tags"),
        ("artist", "tags"),
        ("track", "genres"),
    ]

    CATEGORICAL_COLUMNS = [
        ("track", "genre_top"),
        ("album", "type"),
        ("album", "information"),
        ("artist", "bio"),
    ]

    EXTRA_COLUMNS = [
        ("album", "engineer"),
        ("album", "producer"),
        ("album", "title"),
        ("artist", "associated_labels"),
        ("artist", "members"),
        ("artist", "name"),
        ("track", "composer"),
        ("track", "information"),
        ("track", "lyricist"),
        ("track", "publisher"),
        ("track", "title"),
    ]

    def process_tracks(
        self, chunk: pd.DataFrame, genre_titles: pd.Series
    ) -> pd.DataFrame:
        """
        Process dataframe before insertion in DB.
        """

        # Process data as in https://github.com/mdeff/fma/utils.py
        for column in self.TAGS_GENRES_COLUMNS:
            chunk[column] = chunk[column].map(ast.literal_eval)

        for column in self.CATEGORICAL_COLUMNS:
            chunk[column] = chunk[column].astype("str")

        # Concatenate tags
        for group in chunk.columns.levels[0]:
            if "tags" in chunk[group].columns:
                chunk.loc[:, (group, "tags")] = chunk.loc[:, (group, "tags")].apply(
                    lambda x: ", ".join(x)
                )

            for html_field in ("information", "bio"):
                if html_field in chunk[group].columns:
                    chunk[group, html_field] = chunk[group, html_field].apply(
                        self.strip_html
                    )

        # Sorting to avoid past-lex sort-depth issues
        subset = chunk[
            self.TAGS_GENRES_COLUMNS + self.CATEGORICAL_COLUMNS + self.EXTRA_COLUMNS
        ].sort_index()

        # Explode dataset based on genres
        exploded = subset.loc[:, "track"].explode("genres")

        # Implode it and update genres
        result = pd.merge(
            exploded.reset_index(),
            genre_titles,
            left_on="genres",
            right_on="genre_id",
            how="left",
        )

        subset.loc[:, ("track", "genres")] = result.groupby("track_id").agg(
            {"genre_title": lambda x: ", ".join([str(obj) for obj in x])}
        )["genre_title"]

        return subset

    @staticmethod
    def process_genres(chunk: pd.DataFrame) -> pd.Series:
        return chunk.rename(columns={"title": "genre_title"})["genre_title"]

    @staticmethod
    def create_encoder_input(chunk: pd.DataFrame) -> pd.Series:
        """
        Turn a single chunk into an embedding input.
        """

        # Building embedding from concatenation of str columns
        return chunk.fillna("").apply(lambda x: " ".join(x), axis=1)

    @staticmethod
    def create_embedding_frame(track_ids: pd.Index, embedding_array: npt.NDArray):
        frame = pd.DataFrame(embedding_array, index=track_ids)
        frame["vector"] = list(frame.values)
        return frame["vector"].reset_index()

    @staticmethod
    def strip_html(text):
        if pd.isna(text):
            return text

        # Remove HTML tags & entities
        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"&[a-zA-Z]+;", "", text)
        return text


class VectorSearch:
    def __init__(self, model_name: str, chunksize: int = CHUNK_SIZE):
        self.model = SentenceTransformer(model_name)

        self.client = DuckDBClient()
        self.client.migrate()

        self.processor = DataProcessor()

        # Automatically adjust to CPU/GPU configuration available
        self.pool = self.model.start_multi_process_pool()

        self.chunksize = chunksize  # Pandas chunk size
        self.model_chunksize = 50  # SentenceTransformer chunk size

    def encode_database(
        self,
        track_filepath: str = "data/tracks.csv",
        genre_filepath: str = "data/genres.csv",
    ):
        genre_titles = self.processor.process_genres(
            pd.read_csv(genre_filepath, index_col=0)
        )
        track_generator = pd.read_csv(
            track_filepath, index_col=0, header=[0, 1], chunksize=self.chunksize
        )

        for chunk in tqdm(track_generator):
            processed = self.processor.process_tracks(chunk, genre_titles)
            encoder_input = self.processor.create_encoder_input(processed)

            # Compute the embeddings using multiprocess pool
            embeddings = self.model.encode_multi_process(
                encoder_input.tolist(), self.pool, chunk_size=self.model_chunksize
            )
            embedding_frame = self.processor.create_embedding_frame(
                processed.index, embeddings
            )

            # Load these in DuckDB
            self.client.insert_tracks(processed)
            self.client.insert_embeddings(embedding_frame)

        self.model.stop_multi_process_pool(self.pool)
        self.client.close()

    def search(self, query_input: str, n_results: int = 5):
        """
        Return k best results for input query.
        """
        embedding = self.model.encode_multi_process(
            [query_input], self.pool, chunk_size=self.model_chunksize
        )
        return self.client.get_nearest_tracks_info(embedding.ravel(), n_results)


if __name__ == "__main__":
    v = VectorSearch("all-MiniLM-L6-v2", chunksize=10_000)
    v.search("Funk")
