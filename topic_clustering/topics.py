import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers.util import cos_sim


class Topics:
    def __init__(
        self,
        dataset,
        clusters,
        n_topics=None,
        stop_words_path="data/polish_stopwords.txt",
    ):
        self.data = dataset.load_data()
        self.clusters = clusters
        self.stop_words = self._get_stop_words(stop_words_path)
        self.docs_in_topic = self._docs_in_topic()
        self._words, self.ctfidf = self.calculate_cTFIDF()

        if n_topics and n_topics > 0:
            self.topics_reduction(n_topics)

    def _docs_in_topic(self):
        grouped = {}
        for t, d in zip(self.clusters, self.data):
            if grouped.get(t):
                grouped[t].append(d)
            else:
                grouped[t] = [d]
        return dict(sorted(grouped.items()))

    @staticmethod
    def _get_stop_words(stop_words_path):
        with open(stop_words_path, "r") as f:
            return [line.rstrip("\n") for line in f.readlines()]

    def calculate_cTFIDF(self):
        # tf [token / token count in topic] * idf [ln( number of docs / token count in docs )]
        topics = [" ".join(topic) for topic in self.docs_in_topic.values()]
        cv = CountVectorizer(stop_words=self.stop_words).fit(topics)

        ttm = cv.transform(topics).toarray()  # topic-term matrix
        tf = (ttm.T / ttm.sum(axis=1)).T  # term frequency
        m = len(self.data)  # docs number
        td = ttm.sum(axis=0)  # token count in all docs
        idf = np.log(m / td)  # inverse document frequency

        return cv.get_feature_names(), tf * idf

    def top_n_words(self, n, topic_number=None):
        if topic_number is None:
            top_n_indices = self.ctfidf.argsort(axis=1)[:, -n:]
            return [
                (
                    list(self.docs_in_topic.keys())[t],
                    [self._words[w] for w in top_n_indices[t, :][::-1]],
                )
                for t in range(len(self.docs_in_topic))
            ]
        else:
            top_n_indices = self.ctfidf.argsort(axis=1)[topic_number + 1, -n:]
            return (topic_number, [self._words[w] for w in top_n_indices[::-1]])

    def count_documents(self, topic_number=None):
        if topic_number is None:
            return [(t, len(d)) for t, d in self.docs_in_topic.items()]
        else:
            return len(self.docs_in_topic.get(topic_number, []))

    def topics_reduction(self, desired_number):
        if desired_number >= len(self.docs_in_topic) - 1:
            pass
        else:
            print(f"Topics found: {len(self.docs_in_topic) - 1}. Compressing...")
            topics_to_merge = len(self.docs_in_topic) - desired_number - 1
            for _ in range(topics_to_merge):
                # cosine similarities of topics
                similarities = cos_sim(self.ctfidf, self.ctfidf).numpy()
                np.fill_diagonal(similarities, 0)

                # merge topics
                topics_no_other = self.count_documents()
                del topics_no_other[-1]
                topic_to_merge = sorted(topics_no_other, key=lambda x: x[1])[0][
                    0
                ]  # smallest
                topic_to_merge_into = (
                    similarities[topics_to_merge].argmax() - 1
                )  # most similar
                self.docs_in_topic[topic_to_merge_into] = (
                    self.docs_in_topic[topic_to_merge_into]
                    + self.docs_in_topic[topic_to_merge]
                )
                del self.docs_in_topic[topic_to_merge]
                self.docs_in_topic = {
                    k - 1: v
                    for k, v in zip(
                        range(len(self.docs_in_topic)),
                        self.docs_in_topic.values(),
                    )
                }

                # calculate new ctfidf
                self._words, self.ctfidf = self.calculate_cTFIDF()
            print(f"Final number of topics: {len(self.docs_in_topic) - 1}")

    def describe_topics(self, n, topic_number=None):
        topics = (
            self.top_n_words(n, topic_number)
            if topic_number is None
            else [0, self.top_n_words(n, topic_number)]
        )
        for topic, words in topics[1:]:
            print("-" * 50)
            print(f"Topic {topic} ({self.count_documents(topic)} documents):")
            for word in words:
                print(f"\t* {word}")
        print("-" * 50)
