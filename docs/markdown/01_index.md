<div style="text-align: center">

<h1> Wordy Rappinghood </h1>
<p>An exploration of LLMs for vector search on music metadata.</p>
<p><i>Alain-Parviz Soltani</i></p>
<img src="docs/images/clare_dudeney_lost_words.jpeg" alt="Clare Dudeney, Lost Words, 2023" width=80%/>

<em style="color: grey"><caption>Clare Dudeney, Lost Words, 2023</caption></em>
</div>

*"Mots pressés, mots sensés, mots qui disent la vérité, mots maudits, mots mentis, mots qui manquent le fruit
d'esprit"*.

Those prescient words about large language models' achievements are not the result of a savvy and
thorough technical analysis; they're taken from 1981 new-wave-hit "Wordy Rappinghood" by Tom Tom Club (founded by
Tina Weymouth et Chris Frantz who, along with David Byrne, formed Talking Heads; I reckon that's another great tagline
for LLMs —
but I digress.)

Today, we're gonna stay in the music realm, and observe what LLMs are capable of when it comes to semantic search and
information retrieval.

# Table of Contents
1. [Our dataset](#our-dataset)
2. [Vectors floating through space](#vectors-floating-through-space)
3. [Loading up the data](#loading-up-the-data)
4. [Taking songs somewhere new](#taking-songs-somewhere-new)

# Our dataset

We'll work with the Free Music Archive (FMA) repository. It's an open source dataset useful on tasks such as information
retrieval; let's dig into the data.

From the creators of the dataset themselves, here's a brief description of the data we'll use:

- `tracks.csv`: per track metadata such as ID, title, artist, genres, tags and play counts, for all 106,574 tracks.
- `genres.csv`: all 163 genres with name and parent (that we won't use today).

There's much more available here, including audio features computed with `librosa`, but we'll stick to text for now.

By simpling glancing at the data, we can see that some text fields are sparsely filled (something that theoretically
shouldn't be an issue with semantic search), and that some genres are decidedly more represented than others.

<div style="text-align: center">
<img src="docs/figures/descriptive_plots.png" alt="Missing values & genre frequency" width=100%/>
</div>