# SOURCE: https://github.com/LinkedInLearning/advanced-nlp-with-python-for-machine-learning-3807097

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import pipeline, DistilBertModel, DistilBertTokenizer
import torch
from wordcloud import WordCloud

# load csv file
df = pd.read_csv("ch4_feedback_data.csv", header=None)
print(f"df.head(): {df.head()}")

# load pre-trained DistilBERT model and tokenizer
pretrained_model_name = "distilbert-base-uncased"
print(f"Loading model and tokenizer: {pretrained_model_name}")
model = DistilBertModel.from_pretrained(pretrained_model_name)
tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model_name)

# determine GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Load DistilBERT pipeline
print(f"Loading pipeline using device: {device}")
sentiment_analysis_bert = pipeline("sentiment-analysis",
                                   model="nlptown/bert-base-multilingual-uncased-sentiment",
                                   device=device)

# Iterate over the rows performing analysis
results = []
for index, row in df.iterrows():
    text = row[0]
    result = sentiment_analysis_bert(text)[0]
    sentiment_label = result["label"]
    sentiment_score = result["score"]
    print(f"text: {text}")
    print(f"sentiment score: {sentiment_score}")
    print(f"sentiment label: {sentiment_label}")
    print("-----")
    result_dict = {
        "Text": text,
        "Sentiment Score": sentiment_score,
        "Sentiment Label": sentiment_label,
    }
    results.append(result_dict)

# turn results into a dataframe
results_df = pd.DataFrame(results)

# save to file
results_filename = "sentiment_analysis_results.csv"
print(f"Saving file to {results_filename}")
results_df.to_csv(results_filename, index=False)

# Word Cloud

# combine text into one string
text_combined = " ".join(results_df["Text"])

wordcloud = WordCloud(
    width=800,
    height=480,
    background_color="white",
    colormap="viridis",
).generate(text_combined)

plot_filename = "sentiment_analysis_wordcloud.png"
print(f"Saving wordcloud to {plot_filename}")
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud from Sentiment Analysis Text")
plt.savefig(plot_filename)

# Pie Chart

label_counts = results_df["Sentiment Label"].value_counts()

plt.figure(figsize=(8, 6))
plt.pie(
    label_counts,
    labels=label_counts.index,
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Distribution of Sentiment Labels")
plot_filename = "sentiment_analysis_pie.png"
print(f"Saving distribution chart to {plot_filename}")
plt.savefig(plot_filename)

# Bar Chart
title = "Bar Chart of Sentiment Labels"
plot_filename = "sentiment_analysis_bar.png"
plt.figure(figsize=(10,6))
label_counts.plot(kind="bar", stacked=True)
plt.title(title)
plt.xlabel("Sentiment Labels")
plt.ylabel("Frequency")

print(f"Saving {title} to {plot_filename}")
plt.savefig(plot_filename)