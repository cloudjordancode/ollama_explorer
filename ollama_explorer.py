"""
L1 — Running Local LLMs with Ollama: Explorer  (STARTER)
==========================================================
Run with:
    python ollama_explorer.py

Prerequisites: Ollama running (`ollama serve`) with a model pulled.
               Adjust MODEL below to match what you have downloaded.

Your goal: run 4 experiments that show how system prompts, context,
           question length, and temperature affect LLM responses.

Experiments:
    1. Same question, 3 different system prompts
    2. RAG-style context grounding (answerable vs. unanswerable question)
    3. Response timing across short / medium / long questions
    4. (Bonus) Temperature comparison: 0.1 vs 1.0

Key concepts:
    Ollama /api/chat endpoint:
        POST http://localhost:11434/api/chat
        Body: {"model": MODEL, "messages": [...], "stream": False}
        Response: {"message": {"content": "..."}, ...}

    Message format:
        {"role": "system", "content": "..."}   — optional system prompt
        {"role": "user",   "content": "..."}   — user question

    Measuring time:
        import time
        start = time.time()
        # ... call generate() ...
        elapsed = time.time() - start
"""

import requests
import time

OLLAMA_URL = "http://localhost:11434"
MODEL = "llama3.2:1b"  # Change to match your downloaded model


# ── Core helper ────────────────────────────────────────────────────────────

def generate(messages: list[dict], temperature: float = 0.7) -> tuple[str, float]:
    """
    Call Ollama /api/chat and return (response_text, elapsed_seconds).

    TODO:
        1. POST to f"{OLLAMA_URL}/api/chat" with:
               {"model": MODEL, "messages": messages,
                "stream": False, "options": {"temperature": temperature}}
        2. Measure elapsed time with time.time()
        3. Extract and return the response text from:
               response.json()["message"]["content"]
        4. Handle requests.exceptions.ConnectionError — return an error
           string and 0.0 so experiments keep running
    """
    
    # Record when the API request begins.
    start = time.time()

    try:
        # Send conversation messages to the local Ollama model.
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature}
            },
            timeout=120
        )

        # Check for HTTP errors and extract the generated answer.
        response.raise_for_status()
        answer = response.json()["message"]["content"]

        # Calculate the total time spent generating the response.
        elapsed = time.time() - start

        return answer, elapsed

    except requests.exceptions.RequestException as error:
        # Return an error message without stopping the experiments.
        return f"Ollama request failed: {error}", 0.0

# ── Experiment 1: System prompt comparison ─────────────────────────────────

def experiment_1():
    """Same question, three different system prompts."""
    question = "What is an API?"

    # TODO: Define 3 system prompts:
    #   - None (empty string — send only the user message, no system role)
    #   - "Explain like I'm 5 years old."
    #   - "You are a senior software architect. Be technical and precise."
    
    # Compare how different system instructions change the same answer.
    prompts = [
        ("No system prompt", None),
        ("Explain like I'm 5", "Explain like I'm 5 years old."),
        ("Senior architect", "You are a senior software architect. Be technical and precise.")
    ]


    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Same question, different system prompts")
    print("=" * 60)

    for label, system_prompt in prompts:
        # Start with the same user question for every test.
        messages = [{"role": "user", "content": question}]

        # Add system instructions only when provided.
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Generate an answer and measure response time.
        answer, elapsed = generate(messages)

        # Display the results for comparison.
        print(f"\n--- {label} ---")
        print(answer)
        print(f"Response time: {elapsed:.2f} seconds")


# ── Experiment 2: RAG-style context grounding ──────────────────────────────

def experiment_2():
    """Provide context; test answerable vs. unanswerable question."""

    # TODO: Write a short paragraph (3–5 sentences) about any topic you choose
    context = (
    "FastAPI is a Python web framework used to build APIs. "
    "It uses Pydantic for data validation and automatically "
    "generates interactive API documentation at /docs. "
    "FastAPI supports asynchronous request handling."
    )

    answerable_question = "How does FastAPI validate data?"

    unanswerable_question = "Who created Kubernetes?"

    system_prompt = (
        "Answer ONLY from the provided context. "
        "If the context does not contain the answer, say exactly: "
        "'I don't have enough information to answer that.'"
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 2: RAG-style context grounding")
    print("=" * 60)

    for label, question in [
        ("Answerable", answerable_question),
        ("Unanswerable", unanswerable_question),
    ]:
        # TODO: Build messages with system_prompt + context embedded, then question
        # TODO: Call generate(), print label + question + response

        messages = [
            {
                "role": "system",
                "content": system_prompt + "\n\nCONTEXT:\n" + context
            },
            {
                "role": "user",
                "content": question
            }
        ]

        answer, elapsed = generate(messages)

        print(f"\n--- {label} ---")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print(f"Response time: {elapsed:.2f} seconds")


# ── Experiment 3: Response timing ──────────────────────────────────────────

def experiment_3():
    """Ask questions of varying length, compare response times."""

    # TODO: Define 3 questions — short (~5 words), medium (~20 words), long (~50 words)
    # Compare response times using prompts of different lengths.
    questions = [
        ("Short", "What is Python?"),
        (
            "Medium",
            "Explain what Python is, describe two common uses, "
            "and identify one reason beginners often choose it "
            "as their first programming language."
        ),
        (
            "Long",
            "Explain the Python programming language, including "
            "its main features, common applications in web development, "
            "data analysis, and artificial intelligence. Describe "
            "how Python compares to other programming languages, "
            "why its ecosystem is popular, and what challenges "
            "developers might encounter when building larger applications."
        )
    ]

    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Response timing")
    print("=" * 60)

    for label, question in questions:
        # TODO: Call generate() (no system prompt needed), print label + time
        
        # Send each question to Ollama and measure generation time.
        messages = [
            {"role": "user", "content": question}
        ]

        answer, elapsed = generate(messages)

        # Display the question length and response time.
        print(f"\n--- {label} Question ---")
        print(f"Prompt length: {len(question.split())} words")
        print(f"Response time: {elapsed:.2f} seconds")
        print(f"Response: {answer}")


# ── Experiment 4 (bonus): Temperature comparison ───────────────────────────

def experiment_4():
    """Same prompt, temperature 0.1 vs 1.0."""
    question = "Tell me a one-sentence fact about the ocean."

    print("\n" + "=" * 60)
    print("EXPERIMENT 4 (Bonus): Temperature comparison")
    print("=" * 60)

    for temp in [0.1, 1.0]:
        # TODO: Call generate() with the given temperature, print temp + response
            # Compare low and high temperature using the same question.
        messages = [
            {"role": "user", "content": question}
        ]

        answer, elapsed = generate(
            messages,
            temperature=temp
        )

        # Display each temperature, response, and generation time.
        print(f"\n--- Temperature: {temp} ---")
        print(f"Response: {answer}")
        print(f"Response time: {elapsed:.2f} seconds")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    experiment_1()
        # Observation: All three prompts explained APIs differently.
        # No system prompt gave a broad explanation, the child-friendly
        # prompt used simple analogies, and the senior architect prompt
        # provided more technical details. System prompts influence
        # how the LLM structures and presents its responses.
    experiment_2()
        # Observation: Ollama correctly answered the FastAPI question
        # using the supplied context and refused the unrelated
        # Kubernetes question due to insufficient information.
        # However, the FastAPI response included additional details
        # beyond the context, showing that grounding instructions
        # do not completely prevent the model from adding information.
    experiment_3()
        # Observation: The medium prompt (6.91s) was faster than the short prompt (7.82s), 
        # while the long prompt took the most time (11.96s). Prompt length alone does not
        # determine response time.
    experiment_4()
        # Observation: Both temperatures produced factual answers.
        # Temperature 0.1 gave a general explanation (0.53s),
        # while 1.0 included a specific example (0.40s).
        # Higher temperature can increase variation, but a single
        # test does not establish a consistent difference.
