"""Skills taxonomy and entity extractor for fast keyword matching."""

from __future__ import annotations

import re
from collections import Counter, OrderedDict

SKILLS_DICT: dict[str, list[str]] = {
    "Python": ["python"],
    "Java": ["java"],
    "C++": ["c++", "cpp"],
    "C#": ["c#", "c sharp"],
    "JavaScript": ["javascript", "js"],
    "TypeScript": ["typescript", "ts"],
    "SQL": ["sql"],
    "Go": ["golang", "go"],
    "PyTorch": ["pytorch"],
    "TensorFlow": ["tensorflow"],
    "Scikit-learn": ["scikit-learn", "sklearn"],
    "React": ["react", "reactjs", "react.js"],
    "Node.js": ["node.js", "nodejs"],
    "Django": ["django"],
    "Flask": ["flask"],
    "FastAPI": ["fastapi"],
    "Spring Boot": ["spring boot", "springboot"],
    "Machine Learning": ["machine learning", "học máy"],
    "Deep Learning": ["deep learning", "học sâu"],
    "NLP": ["nlp", "natural language processing", "xử lý ngôn ngữ tự nhiên"],
    "Computer Vision": ["computer vision", "thị giác máy tính"],
    "Data Analysis": ["data analysis", "phân tích dữ liệu"],
    "MySQL": ["mysql"],
    "PostgreSQL": ["postgresql", "postgres"],
    "MongoDB": ["mongodb"],
    "Redis": ["redis"],
    "Docker": ["docker"],
    "Kubernetes": ["kubernetes", "k8s"],
    "Git": ["git"],
    "AWS": ["aws", "amazon web services"],
    "GCP": ["gcp", "google cloud"],
    "CI/CD": ["ci/cd", "ci-cd"],
}


def find_entities(text: str, taxonomy: dict[str, list[str]]) -> OrderedDict[str, int]:
    """Find matches of taxonomy aliases in text and return canonical names ordered by appearance."""
    if not text:
        return OrderedDict()
    text_lower = text.casefold()
    found: list[str] = []
    for canonical, aliases in taxonomy.items():
        for alias in aliases:
            pattern = rf"(?<![\w#+.]){re.escape(alias.casefold())}(?![\w#+.])"
            if re.search(pattern, text_lower):
                found.append(canonical)
                break

    counts = Counter(found)
    ordered: OrderedDict[str, int] = OrderedDict()
    for item in found:
        if item not in ordered:
            ordered[item] = counts[item]
    return ordered
