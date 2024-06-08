# Vectors floating through space

> <div style="color: grey"><i>Recommended listen: [Lonnie Liston Smith – Floating Through Space (Loveland, 1978)](https://youtu.be/_kmHS-IndZw?feature=shared)</i></div>

The goal of vector search — as opposed to more traditional lexical search — is to compare textual objects in a larger
dimensional space, where non-intuitive connections and similarities could be found.

In this new space, items are represented by vectors, called *embeddings*; given a good mapping towards this
high-dimensional space and a meaningful distance, we could find relevant objects by picking the closest neighbours to
our query input.

We'll use cosine similarity to compare our embeddings, $T$ being defined as the target embedding, and $Q$ the query
input counterpart:

$$
S_C (T, Q) = \cos(\theta) = {\mathbf{T} \cdot \mathbf{Q} \over \|\mathbf{T}\| \|\mathbf{Q}\|} = \frac{ \sum\limits_
{i=1}^{n}{T_i Q_i} }{ \sqrt{\sum\limits_{i=1}^{n}{T_i^2}} \cdot \sqrt{\sum\limits_{i=1}^{n}{Q_i^2}} }
$$

Notably, this computes the angle between the two vectors, and disregards magnitude: this is important, as clear-cut,
short content can be as semantically meaningful as a lengthy piece of prose describing the same concept.

Embeddings are generally produced by the encoder block of
a [Transformer](https://huggingface.co/learn/nlp-course/en/chapter1/4); for text data, we'll use BERT-based Transformers
that turn sentences into vector embeddings.

`SentenceTransformers` is a great library, built on top of Hugging Face's `transformers`, that gives us access to many
classical pre-trained models.
In this blog post, we'll use `all-MiniLM-L6-v2`, which offers a great trade-off
between [semantic performance and model size](https://sbert.net/docs/sentence_transformer/pretrained_models.html).

# Loading up the data

> <div style="color: grey"><i>Recommended listen: [NOW Ensemble – City Boy (Dreamfall, 2015)](https://youtu.be/_kmHS-IndZw?feature=shared)</i></div>

Let's now build our vector search database.

First up, we'll clean up the `tracks` data — attributing genres by joining the aforementioned table, and stripping
unwanted HTML attributes — and we'll load it up into a database capable of performing vector search similarity (VSS).

Some databases already offer such features (e.g. `pg_vector` for PostgreSQL), while others are being implemented (see
Alex
Garcia's [wonderful SQLite extension](https://alexgarcia.xyz/blog/2024/building-new-vector-search-sqlite/index.html)).
Here, I used DuckDB, which recently released an experimental extension aptly-named `vss`, capable of performing semantic
search by way
of [Hierarchical Navigable Small World (HNSW) graphs](https://www.pinecone.io/learn/series/faiss/hnsw/), which offers
great efficiency in such case.

Speaking of optimizations, we'll try to be mindful of memory and throughput as much as we can here. We'll send Pandas
data in chunks through a generator to control memory usage, and make use of DuckDB's optimizations to preserve
row-to-row overhead, such as reading from JSON objects.

To create the vector table, the SQL initial migration looks something like this:

```
CREATE TABLE IF NOT EXISTS embeddings (
    track_id INTEGER PRIMARY KEY,
    vector FLOAT[N_EMBED] -- embedding size: depends on your model
);

/* Manually load vector similarity search.
Allow persistence of HNSW index to use outside of in-memory dbs.*/
INSTALL vss;
LOAD vss;
SET hnsw_enable_experimental_persistence = TRUE;
CREATE INDEX IF NOT EXISTS cos_index ON embeddings USING HNSW (vector) 
WITH (metric = 'cosine');
```

(**Note**: WAL recovery hasn't been properly implemented as of now for HNSW-indexed tables, hence the persistence flag;
you might want to wait a little before putting this into production.)

We'll store both the structured and embedded data in two different tables (`tracks` and `embeddings`).
As encoding our embeddings amounts to a forward pass on the encoder portion of our model, the operation can be a tad
intensive depending on the size of our input.

To distribute encoding across several GPUs/CPUs, we can set a `pool` at the initialization of our process, and make use
of method `start_multi_process_pool`:

```
class VectorSearch:
    def __init__(self, model_name: str, chunksize: int = CHUNK_SIZE):
        ...
        
        self.model = SentenceTransformer(model_name)
        self.pool = self.model.start_multi_process_pool()

    def encode_database(self):
        ...

        for chunk in track_generator:
            # Clean data, and concatenate it as a sentence
            processed = self.processor.process_tracks(chunk, genre_titles)
            encoder_input = self.processor.create_encoder_input(processed)

            # Compute the embeddings
            embeddings = self.model.encode_multi_process(
                encoder_input.tolist(), 
                self.pool, 
                chunk_size=self.model_chunksize
            )
            ...
            
            # Load these in DuckDB
            self.client.insert_tracks(processed.reset_index())
            self.client.insert_embeddings(embedding_frame)

        self.model.stop_multi_process_pool(self.pool)
        self.client.close()
```

To test my solution and make use of GPUs comes encoding time, I deployed a pod on [RunPod](https://www.runpod.io) (no
affiliation to this website) with 2 x RTX
4090 GPUs (at 0.74$/Hr, they're pretty cost-effective), 25 vCPU and 200 GB of RAM.

The entire process takes about 1.1 per batch of 10,000 rows, 12.2 seconds total (`UPSERT` times are of the same order of
magnitude, as we rely on DuckDB data ingestion optimizations).

To put it into perspective, the CPU-only performance on a M-series MacBook Pro with 4P + 4E cores is 98.3 per batch,
or ~18 minutes total. Not bad!

We can now perform vector search on our database:

```
def search(self, query_input: str, n_results: int = 5):
    """
    Return k best results for input query.
    """
    embedding = self.model.encode_multi_process(
        [query_input], self.pool, chunk_size=self.model_chunksize
    )
    return self.client.get_nearest_tracks_info(embedding.ravel(), n_results)
```