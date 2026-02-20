def agent(query):
    if "sentiment" in query.lower():
        return "sentiment"
    elif "topic" in query.lower():
        return "topic"
    else:
        return "rag"
