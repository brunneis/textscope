import json
from textscope.relevance_analyzer import RelevanceAnalyzer
from textscope.subtheme_analyzer import SubthemeAnalyzer
import nltk
nltk.download('punkt')

rel = RelevanceAnalyzer()
subtheme = SubthemeAnalyzer()

with open('x_posts_ciber.jsonl', 'r') as file:
    data = file.readlines()
    for line in data:
        tweet = json.loads(line)
        text = tweet["full_text"]
        score = rel.analyze(text, "cibersec")
        tweet["score_rel"] = score
        puncts = subtheme.analyze_bin(text, "cibersec")
        tweet["subthemes"] = puncts
        print(f'{puncts}')