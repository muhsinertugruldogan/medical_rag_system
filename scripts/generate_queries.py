import random
import csv

SHORT = [
    "Is there cardiomegaly?",
    "Are the lungs clear?",
    "Is pneumothorax present?",
    "Is pleural effusion present?",
    "Is there acute disease?",
    "Are there abnormal findings?",
]

MEDIUM_TEMPLATES = [
    "Based on the chest X-ray, {}",
    "From the radiology report, {}",
    "Considering the findings, {}",
    "Can you determine if {}",
]

MEDIUM_CORE = [
    "is there cardiomegaly?",
    "there is pleural effusion?",
    "the lungs are clear?",
    "there are acute abnormalities?",
    "there is pneumothorax?",
]

LONG = [
    "Based on the provided chest X-ray and retrieved radiology reports, what is the main abnormality observed and is there evidence of acute cardiopulmonary disease?",
    
    "Analyze the chest X-ray and the associated reports. What are the key radiographic findings and do they indicate any acute pathology?",
    
    "Review the findings in the image and reports. Describe the main impression and whether there are acute abnormalities.",
    
    "Considering both the image-derived findings and retrieved reports, summarize the radiographic impression and indicate if any acute disease is present.",
]

TREATMENT = [
    "What is the abnormality and what are the current treatment options?",
    "Describe the findings and suggest possible treatment approaches.",
    "What is the diagnosis and how is it typically treated?",
    "What abnormality is present and what are the recommended treatments?",
]


def generate_queries():
    queries = []

    # 40 SHORT
    for _ in range(40):
        queries.append(random.choice(SHORT))

    # 30 MEDIUM
    for _ in range(30):
        t = random.choice(MEDIUM_TEMPLATES)
        c = random.choice(MEDIUM_CORE)
        queries.append(t.format(c))

    # 20 LONG
    for _ in range(20):
        queries.append(random.choice(LONG))

    # 10 TREATMENT (edge case)
    for _ in range(10):
        queries.append(random.choice(TREATMENT))

    random.shuffle(queries)
    return queries


def save_csv(queries):
    with open("queries.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query", "length"])

        for q in queries:
            length = (
                "short" if len(q.split()) < 8
                else "medium" if len(q.split()) < 18
                else "long"
            )
            writer.writerow([q, length])


if __name__ == "__main__":
    qs = generate_queries()
    save_csv(qs)
    print("queries.csv saved")