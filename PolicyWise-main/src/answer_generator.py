
import os

FALLBACK = "Not specified in the provided policies."


def generate_answer(question, chunks, use_llm=False):

    if not chunks:
        return FALLBACK, []

    citations = list(dict.fromkeys(
        (c["source"], c["page"])
        for c in chunks
    ))

    if use_llm and os.getenv("OPENAI_API_KEY"):

        try:
            from openai import OpenAI

            context = "\n\n".join(
                f"[{c['source']}, page {c['page']}, "
                f"status: {c.get('status', 'unknown')}]\n"
                f"{c['text']}"
                for c in chunks
            )

            response = OpenAI().chat.completions.create(
                model=os.getenv(
                    "OPENAI_MODEL",
                    "gpt-4o-mini"
                ),
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a source-grounded HR policy assistant. "

                            "Answer using ONLY the supplied policy evidence. "

                            "Provide a direct and concise answer to the user's question. "

                            "Do not invent policy rules, numerical values, dates, "
                            "eligibility conditions, or citations. "

                            "Always distinguish current policies from superseded policies. "

                            "For questions about current policies, never use "
                            "superseded policy information as the current rule. "

                            "If a policy applies only to full-time employees, "
                            "do not assume it also applies to contractors. "

                            "If the supplied evidence is insufficient or does not "
                            "answer the question, reply exactly: "
                            f"{FALLBACK} "

                            "Cite the source filename and page number "
                            "when providing a supported answer."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Evidence:\n{context}"
                        )
                    }
                ]
            )

            answer = response.choices[0].message.content

            if not answer:
                return FALLBACK, []

            return answer.strip(), citations

        except Exception as exc:
            return (
                f"Answer generation unavailable. "
                f"Please inspect the retrieved evidence. ({exc})",
                citations
            )

    # Offline mode: display evidence without pretending
    # that an LLM has synthesized an answer.

    evidence_text = "\n\n".join(
        f"{c['text']} ({c['source']}, page {c['page']})"
        for c in chunks
    )

    return (
        "Relevant retrieved policy evidence:\n\n" + evidence_text,
        citations
    )
