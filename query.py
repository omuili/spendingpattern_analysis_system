# query_generator.py
import random
from transaction_data import generate_transactions, categories  # Importing necessary functions and variables

# Generate a small set of transactions to use for query generation
transactions = generate_transactions(1000)

def create_queries(num):
    # Updated query templates
    query_templates = [
        "How much did I spend on {} last month?",
        "Show my transaction history for the last {}.",
        "What's my total spending on {} this year?",
        "List all {} expenses.",
        "What category did I spend mostly on in the past 3 months?",
        "What spending category am I most inconsistent in?",
        "What was my largest {} bill?"
    ]
    time_frames = ["week", "month", "year"]
    queries = []

    for _ in range(num):
        # Randomly select a category from the CATEGORIES dictionary
        category = random.choice(list(categories.keys()))
        time_frame = random.choice(time_frames)

        # Choose a random template and format it with transaction data
        template = random.choice(query_templates)
        if template.count('{}') == 1:
            query = template.format(category)
        elif template.count('{}') == 2:
            query = template.format(category, time_frame)
        else:
            query = template  # If no placeholders, use the template as is.
        queries.append(query)

    return queries
