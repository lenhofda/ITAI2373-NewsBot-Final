
# Module 9 - Topic Modeling with LDA and NMF
# ITAI 2373 - NewsBot Intelligence System 2.0

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


def run_lda(df):
    """Train LDA topic model using processed BBC News articles."""
    count_vectorizer = CountVectorizer(
        max_features=3000,
        stop_words='english',
        min_df=2,
        max_df=0.95
    )

    count_matrix = count_vectorizer.fit_transform(
        df['text_processed'].fillna('')
    )

    lda_model = LatentDirichletAllocation(
        n_components=5,
        random_state=42,
        max_iter=10
    )

    lda_topics = lda_model.fit_transform(count_matrix)

    print("LDA Topic Modeling Complete!")
    print("Number of articles:", count_matrix.shape[0])
    print("Number of topics:", lda_model.n_components)
    print("Topic matrix shape:", lda_topics.shape)

    return lda_model, lda_topics, count_vectorizer


def run_nmf(df):
    """Train NMF topic model using TF-IDF features."""
    nmf_vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        min_df=2,
        max_df=0.95
    )

    nmf_matrix = nmf_vectorizer.fit_transform(
        df['text_processed'].fillna('')
    )

    nmf_model = NMF(
        n_components=5,
        random_state=42,
        init='nndsvda',
        max_iter=400
    )

    nmf_topics = nmf_model.fit_transform(nmf_matrix)

    feature_names_nmf = nmf_vectorizer.get_feature_names_out()

    for topic_idx, topic in enumerate(nmf_model.components_):
        top_indices = topic.argsort()[-10:][::-1]
        top_words = [feature_names_nmf[i] for i in top_indices]

        print(f"\nNMF Topic {topic_idx + 1}:")
        print(", ".join(top_words))

    print("\nNMF Topic Modeling Complete!")

    return nmf_model, nmf_topics, nmf_vectorizer
