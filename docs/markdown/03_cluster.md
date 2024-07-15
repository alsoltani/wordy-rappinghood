# Taking songs somewhere new

> <div style="color: grey"><i>Recommended listen: <a href="https://youtu.be/BG0XBnMyXcQ" target="_blank">Floating Points — Silhouette (I, II & III) (Elaenia, 2015)</a></i></div>

106,574 is a lot of tracks, and we might want a bit of perspective on the set of songs we're looking at.

So far, we've expressed our tracks in a space that allows semantic comparisons; in this representation, data points are
of size 384, the size of the final hidden layer of our encoder — which can be much larger if we use a more powerful
model:
for instance, [NV-Embed-1](https://huggingface.co/nvidia/NV-Embed-v1), the current leader on
the [Massive Text Embedding Benchmark](https://huggingface.co/spaces/mteb/leaderboard) on Hugging Face, has an
output size of 4096. That's a lot of features, and the curse of dimensionality is never far away.

At this step, it is natural to perform _dimensionality reduction_: turn our 384-sized embeddings into vectors of
lower
dimension, while hopefully preserving most of the semantic capabilities we've just gained.

[Uniform Manifold Approximation and Projection (UMAP)](https://arxiv.org/abs/1802.03426) is a good candidate for
this: it's great at preserving local structure,
and [scales much better](https://pair-code.github.io/understanding-umap/) than comparable algorithms like t-SNE.

From there, we can group these into meaningful clusters, using _density-based clustering_: we'll search for areas —
let's called these _islands_ —
of higher density, and anything below a certain level — _sea level_, if you will — will be deemed noise. In practical
terms, we'll get $m$ clusters $C_1, \dots, C_m$ and $C_{m+1} = -1$ will be our background noise.

We'll use the hierarchical variant of the popular non-parametric `DBSCAN`
algorithm [HDBSCAN](https://joss.theoj.org/papers/10.21105/joss.00205), which works well with non-flat geometry and
uneven cluster sizes.
Icing on the cake, both `UMAP` and `HDBSCAN` have GPU-accelerated implementations provided by the RapidsAI `cuML`
library. What's
not to love?

## Our clustering class

The implementation of our clustering class is pretty straighforward, so I'll gloss over the details.
This is inspired by the `DenseClus` library by AWS Labs, which was initially designed to process mixed-type
(categorical + numerical) data, and from which I'll reuse the numerical extracting tools.

```
class Clustering:
    """
    UMAP on textual embeddings + HDBSCAN clustering.
    """
    ...
    def init_models(self, params: Dict[str, Union[float, str]]):
        ...
        # To be called within constructor.
        # If we've got some nice GPUs at hand:
        self.umap = cuUMAP(
            metric="cosine",
            verbose=False,
            **umap_params,
        )
        self.hdbscan = cuHDBSCAN(
            prediction_data=True,
            **hdbscan_params,
        )
            
    def fit(self, vector_embeddings: pd.DataFrame):
        self.umap.fit(vector_embeddings)
        self.embedding = self.umap.embedding_
        self.hdbscan.fit(self.embedding)
        self.labels = self.hdbscan.labels_
```

We now need to tune and ensure the validity of our model, and there are some important
subtleties along the way.

# Fine-tuning our algorithms

> <div style="color: grey"><i>Recommended listen: <a href="https://youtu.be/HtUceMv3wjk" target="_blank">Stereolab — Cybele's Reverie (Emperor Tomato Ketchup, 1996)</a></i></div>

Notoriously, both algorithms are very sensitive to hyperparameter selection:

- UMAP can provide drastically different results depending on:
    - The number of neighbours, `n_neighbors`, controlling the balance between global and local structure preservation
    - The minimal distance that set points apart, `min_dist`
    - The number of dimensions at the end of the embedding, `n_components`.
- HDBSCAN itself is impacted by:
    - The minimal cluster size, `min_cluster_size`
    - The minimal number of samples per cluster, `min_samples`
    - A parameter `cluster_selection_epsilon`, that ensures clusters below such threshold are not split up any further
    - And a final `clustering_selection_method`, indicating how flat clusters are formed from the tree hierarchy.

Most of these parameters are dataset-dependent: how do we want to form clusters? Should we allow for great class
imbalance? These are questions that are highly dependent on domain expertise, but for the
sake of this post, we'll just assume we don't have sufficient domain knowledge to answer these.

Instead, let's frame it as a generic optimization problem: we'll try to find the set of hyperparameters that offers us
the lowest loss on our training set.

# An all-encompassing loss

Contrary to standard classification problems, it's hard to define what accuracy means in a clustering setting, as we
don't have access to any _ground-truth_ labels.

Ideally, we'd like a loss that assesses the quality of both our manifold and our clustering. We want a final model
that:

1. Form manifolds that nicely retain the local structure of our data
2. While providing good separation comes clustering time.

Assuming independence between the two steps of our pipeline, we could devise a composite loss of the following form:
$$l(X) = 1 - Q_{ma}(X, X_e) Q_{cl}(X_e, C)$$

where $X$ is our dataset, $X_e$ the embeddings produced by the manifold, and $C$ the final clusters.
$Q_{ma}, Q_{cl} \in [0, 1]$ respectively measure the quality of our embeddings and the clustering itself; these two
quantities ought to be maximized by the grid search procedure.

## Embedding Quality

Let's first introduce the embedding-specific portion of our loss: the _trustworthiness_ of our manifold.

Is is defined as[^1]

$$
T(X, X_e, k) =\frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
\sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))
$$

Intuitively, this looks at the $k$ nearest neighbours in the output space, and penalizes any unexpected elements based
on their rank $r(i, j)$.
Thus a high trustworthiness will ensure you can preserve local information correctly.

[^1]: For each sample $i$, $\mathcal{N}_{i}^{k}$
are its $k$ nearest neighbors in the output space, and every sample $j$ is its $r(i, j)$-th
nearest neighbor in the input space ($r$ being the rank function in distance between samples $i$ and $j$).

## Clustering Quality

To assess the quality of our clustering, we'll use the _validity index_ defined as
$$
D(X_e, C) \in [-1, 1]
$$

It comes from [this paper](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf) and is often referred to as
Density-Based Clustering Validity.

It tries to assign high scores to clusters where:

- Points within each cluster are densely packed (_low core distances_)
- Different clusters are well-separated from each other (_high separation_).

It's a fairly standard evaluation metric of density-based clustering techniques[^2]. We'll normalize it between 0 and 1.

[^2]: Classical clustering scores based on a measure of distance between the clustered observations, such as
Silhouette, don't work in a density-based setting, which is the main reason behind the usage of DBCV. That being said,
an extension of Silhouette to such cases has been proposed
in [this paper](https://link.springer.com/article/10.1007/s11222-010-9169-0).

## Composite Loss

Our final composite loss looks like this:
$$l(X, k) = 1 - T(X, X_e, k) \tilde{D}(X_e, C)$$ with $\tilde{D}$ the normalized DBCV index.

HDBSCAN has a tendency to form one of two huge clusters, and a collection of tiny ones afterward (in relative size).
This can be alleviated by preferring `leaf` to `eom` ('Excess of Mass') as a clustering selection method. 

But from an optimization perspective, we can also add a penalization term to coerce our model into producing more homogenous clusters; we'll use an entropy-based penalty:

$$l(X, k) = 1 - T(X, X_e, k) \tilde{D}(X_e, C) + \alpha (1 - S(X))$$

where

$$ S(X) = - \sum_i \frac{p_i \log_2 p_i}{\log_2 n}$$ is the normalized Shannon entropy, taken on the output
probabilities of our model $p$.

Just like $\ell_1$-regularization can be seen as MAP estimation of a linear regression model with a Laplace prior, this
is like choosing a prior based on [Jaynes' principle of maximum entropy](https://caseychu.io/posts/maximum-entropy-kl-divergence-and-bayesian-inference/).

We'll use this loss to perform our hyperparameter optimization on.