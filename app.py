from flask import Flask, render_template, request, redirect, url_for, flash
import sqlite3
import re
from datetime import datetime
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "interviews.db"

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-this-secret-key"


QUESTIONS = [
    {
        "id": 1,
        "role": "Software Developer",
        "difficulty": "Beginner",
        "question": "What is a variable in programming?",
        "keywords": ["variable", "value", "memory", "data", "store"],
        "ideal": "A variable is a named location or reference used to store a value in memory. The value can usually be changed while a program runs."
    },
    {
        "id": 2,
        "role": "Software Developer",
        "difficulty": "Beginner",
        "question": "What is the difference between a list and a tuple in Python?",
        "keywords": ["list", "tuple", "mutable", "immutable", "change"],
        "ideal": "A Python list is mutable, so its elements can be changed after creation. A tuple is immutable, so its elements cannot be changed after creation."
    },
    {
        "id": 3,
        "role": "Software Developer",
        "difficulty": "Beginner",
        "question": "What is an algorithm?",
        "keywords": ["algorithm", "steps", "problem", "solution", "finite"],
        "ideal": "An algorithm is a finite sequence of clear steps used to solve a problem or perform a computation."
    },
    {
        "id": 4,
        "role": "AI / ML Engineer",
        "difficulty": "Beginner",
        "question": "What is machine learning?",
        "keywords": ["machine", "learning", "data", "model", "prediction", "pattern"],
        "ideal": "Machine learning is a field of AI where algorithms learn patterns from data and use those patterns to make predictions or decisions without being explicitly programmed for every case."
    },
    {
        "id": 5,
        "role": "AI / ML Engineer",
        "difficulty": "Beginner",
        "question": "What is the difference between supervised and unsupervised learning?",
        "keywords": ["supervised", "unsupervised", "label", "data", "classification", "clustering"],
        "ideal": "Supervised learning uses labeled data to learn a mapping from inputs to outputs, while unsupervised learning works with unlabeled data to discover patterns such as clusters."
    },
    {
        "id": 6,
        "role": "AI / ML Engineer",
        "difficulty": "Beginner",
        "question": "What is overfitting in machine learning?",
        "keywords": ["overfitting", "training", "test", "generalize", "noise", "model"],
        "ideal": "Overfitting happens when a model learns the training data too closely, including noise, and therefore performs poorly on unseen data."
    },
    {
        "id": 7,
        "role": "Data Analyst",
        "difficulty": "Beginner",
        "question": "What is the difference between mean and median?",
        "keywords": ["mean", "median", "average", "middle", "data", "outlier"],
        "ideal": "The mean is calculated by adding all values and dividing by the number of values. The median is the middle value after sorting the data and is less affected by outliers."
    },
    {
        "id": 8,
        "role": "Data Analyst",
        "difficulty": "Beginner",
        "question": "Why is data cleaning important?",
        "keywords": ["data", "cleaning", "missing", "duplicate", "error", "quality"],
        "ideal": "Data cleaning improves data quality by handling missing values, duplicates, incorrect formats, and errors so that analysis and models produce more reliable results."
    },
    {
        "id": 9,
        "role": "General / HR",
        "difficulty": "Beginner",
        "question": "Tell me about yourself.",
        "keywords": ["education", "skills", "project", "experience", "goal"],
        "ideal": "Give a concise introduction covering your education or current role, relevant skills, one or two projects or experiences, and the type of opportunity you are seeking."
    },
    {
        "id": 10,
        "role": "General / HR",
        "difficulty": "Beginner",
        "question": "Why should we hire you?",
        "keywords": ["skills", "learn", "contribute", "team", "problem", "growth"],
        "ideal": "Explain the relevant skills you bring, evidence from projects or experience, your ability to learn, and how you can contribute to the team."
    },
]


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                score INTEGER NOT NULL,
                answered INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                score INTEGER NOT NULL,
                feedback TEXT NOT NULL,
                FOREIGN KEY(interview_id) REFERENCES interviews(id)
            )
        """)


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())).strip()


def evaluate_answer(answer, question):
    answer = answer.strip()
    if not answer:
        return 0, "No answer was provided. Try answering in 2–5 clear sentences."

    clean_answer = normalize(answer)
    clean_ideal = normalize(question["ideal"])

    # Keyword coverage
    words = set(clean_answer.split())
    keyword_hits = sum(1 for keyword in question["keywords"] if keyword in words)
    keyword_score = (keyword_hits / len(question["keywords"])) * 50

    # Semantic similarity using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([clean_answer, clean_ideal])
    similarity = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    similarity_score = similarity * 40

    # Structure / completeness signal
    sentence_count = max(1, len(re.findall(r"[.!?]+", answer)))
    length_score = min(len(answer.split()) / 35, 1.0) * 10
    if sentence_count >= 2:
        length_score += 2
    length_score = min(length_score, 10)

    score = round(min(100, keyword_score + similarity_score + length_score))

    if score >= 80:
        feedback = "Excellent answer. It is relevant and covers the main concepts."
    elif score >= 65:
        feedback = "Good answer. Add one concrete example or explain the key concept in more detail."
    elif score >= 45:
        feedback = "Fair answer. Cover more of the important concepts and make the explanation more structured."
    else:
        feedback = "Needs improvement. Start with a definition, explain the main idea, and give a simple example."

    return score, feedback


def get_questions(role, difficulty):
    return [
        q for q in QUESTIONS
        if q["role"] == role and q["difficulty"] == difficulty
    ]


@app.route("/")
def home():
    roles = sorted(set(q["role"] for q in QUESTIONS))
    difficulties = sorted(set(q["difficulty"] for q in QUESTIONS))
    return render_template("index.html", roles=roles, difficulties=difficulties)


@app.route("/start", methods=["POST"])
def start():
    role = request.form.get("role", "").strip()
    difficulty = request.form.get("difficulty", "").strip()

    questions = get_questions(role, difficulty)

    if not questions:
        flash("No questions are available for that combination yet.")
        return redirect(url_for("home"))

    return render_template(
        "interview.html",
        role=role,
        difficulty=difficulty,
        questions=questions,
        total=len(questions),
    )


@app.route("/submit", methods=["POST"])
def submit():
    role = request.form.get("role", "").strip()
    difficulty = request.form.get("difficulty", "").strip()
    question_ids = request.form.getlist("question_id")

    questions_by_id = {str(q["id"]): q for q in QUESTIONS}
    results = []
    total_score = 0
    answered = 0

    for qid in question_ids:
        question = questions_by_id.get(qid)
        if not question:
            continue

        answer = request.form.get(f"answer_{qid}", "").strip()
        score, feedback = evaluate_answer(answer, question)
        if answer:
            answered += 1

        total_score += score
        results.append({
            "question": question["question"],
            "answer": answer,
            "score": score,
            "feedback": feedback,
        })

    overall = round(total_score / len(results)) if results else 0

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO interviews
            (role, difficulty, score, answered, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                role,
                difficulty,
                overall,
                answered,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        interview_id = cursor.lastrowid

        for result in results:
            conn.execute(
                """
                INSERT INTO answers
                (interview_id, question, answer, score, feedback)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    interview_id,
                    result["question"],
                    result["answer"],
                    result["score"],
                    result["feedback"],
                ),
            )

    return render_template(
        "results.html",
        role=role,
        difficulty=difficulty,
        results=results,
        overall=overall,
        answered=answered,
        total=len(results),
    )


@app.route("/history")
def history():
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT * FROM interviews
            ORDER BY id DESC
            LIMIT 20
            """
        ).fetchall()

    return render_template("history.html", interviews=rows)


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
