from src.data_loader import load_data
from src.preprocessing import clean_text
from src.sentiment import train_sentiment, sentiment_summary
from src.topic_model import get_topics
from src.embeddings import create_embeddings
from src.vector_store import build_faiss
from src.rag import retrieve
from src.agent import agent
from src.report_generator import generate_report

# STEP 1
df = load_data("data/sentiment140.csv" , keyword="") 
df["clean"] = df["text"].apply(clean_text)

print(df["sentiment"].unique())
print(df["sentiment"].isnull().sum())
print(df.shape)

# STEP 2 Sentiment
model, vectorizer = train_sentiment(df["clean"], df["sentiment"])
sent_stats = sentiment_summary(model, vectorizer, df["clean"])

# STEP 3 Topics
topics = get_topics(df["clean"])

# STEP 4 Embeddings + FAISS
embed_model, embeddings = create_embeddings(df["clean"].tolist())
index = build_faiss(embeddings)

# STEP 5 Agent Example
query = "Why are people unhappy about economy?"
tool = agent(query)

if tool == "rag":
    results = retrieve(query, embed_model, index, df["text"].tolist())
    print("Top Retrieved Tweets:", results)

# STEP 6 Report
report = generate_report(sent_stats, topics)
print(report)
