
import re

from ingestion import VectorStore, get_embedder


CURRENT_POLICY_PATTERN = re.compile(
    r"\b(current|currently|latest|most recent|in force|valid now)\b",
    re.IGNORECASE
)

HISTORICAL_POLICY_PATTERN = re.compile(
    r"\b(old|older|previous|historical|superseded|outdated|former)\b",
    re.IGNORECASE
)


class PolicyRetriever:
    """
    Semantic retrieval using the persisted FAISS/NumPy index.

    Current-policy questions exclude superseded documents.
    Historical questions prioritize superseded documents.
    General questions prioritize current documents.
    """

    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, question, k=3):

        if not question or not question.strip():
            return []

        query_vector = self.embedder.encode([question])[0]

        # Retrieve all indexed chunks before applying policy-version rules.
        hits = self.vector_store.search(
            query_vector,
            k=len(self.vector_store.chunks)
        )

        asks_current = bool(
            CURRENT_POLICY_PATTERN.search(question)
        )

        asks_historical = bool(
            HISTORICAL_POLICY_PATTERN.search(question)
        )

        # Explicit current-policy questions must not use
        # superseded policy documents as current evidence.
        if asks_current and not asks_historical:

            hits = [
                (score, chunk)
                for score, chunk in hits
                if chunk.get("status") == "current"
            ]

        elif asks_historical:

            hits.sort(
                key=lambda hit: (
                    hit[1].get("status") != "superseded",
                    -hit[0]
                )
            )

        # For general questions, prioritize current policies.
        else:
            hits.sort(
                key=lambda hit: (
                    hit[1].get("status") != "current",
                    -hit[0]
                )
            )

        results = []

        for score, chunk in hits[:k]:

            item = dict(chunk)

            item["score"] = float(score)
            item["source"] = item["doc_name"]
            item["version"] = item.get("status", "unknown")

            results.append(item)

        return results
