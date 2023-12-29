# Import necessary libraries
from transaction_data import generate_transactions, categories
from query import create_queries
import random
random.seed(42)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, TimeDistributed, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
import numpy as np

# Constants
num_transactions = 1000
num_queries = 500

# Create synthetic transactions and queries
synthetic_transactions = generate_transactions(num_transactions)
synthetic_queries = create_queries(num_queries)


# Load data
transactions = pd.DataFrame(synthetic_transactions)
if isinstance(transactions, pd.DataFrame):
    try:
        transactions['date_time'] = pd.to_datetime(transactions['date_time'])
    except Exception as e:
        print(f"Error converting 'date_time': {e}")

 # list of queries
queries = synthetic_queries

# Label queries with intents and entities
def label_query(query):
    query = query.lower()
    intent = "unknown"
    entities = {}

    # Determine the intent
    if "how much did i spend on" in query:
        intent = "total_spend_category"
    elif "show my transaction history" in query:
        intent = "transaction_history"
    elif "what spending category am i most inconsistent in" in query or "most inconsistent" in query:
        intent = "most_inconsistent_category"
    elif "what's my total spending on" in query:
        intent = "total_spend_category"
    elif "list all" in query:
        intent = "transaction_history"
    elif "what category did i spend mostly on in the past 3 months" in query:
        intent = "most_spent_category_3_months"
    elif "what was my largest" in query:
        intent = "largest_bill_category"
    elif "what category did i spend mostly on in the past 3 months" in query:
        intent = "most_spent_category_3_months"
    else:
        intent = "unknown"

    # Default time frame for certain intents
    if intent == "most_inconsistent_category":
        entities["TIME_FRAME"] = "3 months"  # Default to 3 months if not specified

    # Extract entities (CATEGORY and TIME_FRAME)
    entities = {}
    for category in categories:
        if category.lower() in query:
            entities["CATEGORY"] = category
            break

    for time_frame in ["week", "month", "year", "3 months"]:
        if time_frame in query:
            entities["TIME_FRAME"] = time_frame
            break

    return {"query": query, "intent": intent, "entities": entities}

# Create the BIO tags for each query
def create_bio_tags(query, entities_info):

    # Tokenize the query
    tokens = query.split()
    labels = ['O'] * len(tokens)  # Initialize all tokens as Outside

    for entity_type, entity_value in entities_info.items():
        entity_tokens = entity_value.split()
        for i, token in enumerate(tokens):
            if tokens[i:i + len(entity_tokens)] == entity_tokens:
                labels[i] = f"B-{entity_type}"  # Begin entity
                # If the entity has more than one token, label the remaining tokens as Inside
                labels[i + 1:i + len(entity_tokens)] = [f"I-{entity_type}"] * (len(entity_tokens) - 1)
                break

    return labels

# Label the queries
labeled_queries = [label_query(query) for query in synthetic_queries]

# Tokenize and pad the queries
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts([label['query'] for label in labeled_queries])
sequences = tokenizer.texts_to_sequences([label['query'] for label in labeled_queries])
data = pad_sequences(sequences, maxlen=50)


# Convert intent labels
intent_labels = [label['intent'] for label in labeled_queries]
intent_encoder = LabelEncoder()
intent_labels_encoded = intent_encoder.fit_transform(intent_labels)

# Prepare entity labels for NER
bio_labeled_queries = [create_bio_tags(lq['query'], lq['entities']) for lq in labeled_queries]
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts([label for sublist in bio_labeled_queries for label in sublist])
entity_sequences = [label_tokenizer.texts_to_sequences(labels) for labels in bio_labeled_queries]
padded_sequences = pad_sequences(entity_sequences, maxlen=50, padding='post')
padded_sequences = np.expand_dims(padded_sequences, -1)  # Add an extra dimension for sparse_categorical_crossentropy

# Split the data for both intent classification and NER
X_train_intent, X_test_intent, Y_train_intent, Y_test_intent = train_test_split(data, intent_labels_encoded, test_size=0.4, random_state=42)
X_train_ner, X_test_ner, Y_train_ner, Y_test_ner = train_test_split(data, padded_sequences, test_size=0.4, random_state=42)


# Model parameters
input_dim = 10000
output_dim = 64
input_length = 50
num_labels = len(intent_encoder.classes_)

# L2 regularization factor
l2_reg = 0.001

#Build intent classification model
intent_model = Sequential()
intent_model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))
intent_model.add(Bidirectional(LSTM(64, return_sequences=False, kernel_regularizer=l2(l2_reg))))
intent_model.add(Dropout(0.5))
intent_model.add(Dense(num_labels, activation='softmax'))

#Compile the model
intent_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Build the NER model
max_seq_length = 50  # from your padded sequences
num_tags = len(label_tokenizer.word_index) + 1  # number of unique entity tags
ner_model = Sequential()
ner_model.add(Embedding(input_dim=10000, output_dim=64, input_length=max_seq_length))
ner_model.add(Bidirectional(LSTM(64, return_sequences=True)))
ner_model.add(TimeDistributed(Dense(num_tags, activation='softmax')))
ner_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the intent classification model with the training data and validate using the validation set
intent_model.fit(X_train_intent, Y_train_intent,
          validation_data=(X_test_intent, Y_test_intent),
          epochs=10,
          callbacks=[early_stopping])

# Evaluate the intent classification model
loss, accuracy = intent_model.evaluate(X_test_intent, Y_test_intent)
print(f"Test accuracy for intent classification: {accuracy}")

# Train the NER model
ner_model.fit(X_train_ner, Y_train_ner, batch_size=64, epochs=8, validation_data=(X_test_ner, Y_test_ner))

# Evaluate the NER model
loss, accuracy = ner_model.evaluate(X_test_ner, Y_test_ner)
print(f"Test accuracy for NER: {accuracy}")


# Make predictions on the test data
intent_predictions = intent_model.predict(X_test_intent)

# Make predictions on the test data
ner_predictions = ner_model.predict(X_test_ner)

def analyze_spending_inconsistency(transactions, n_months=3):
    # Check and convert transactions to a DataFrame if it's not already one
    if not isinstance(transactions, pd.DataFrame):
        try:
            transactions = pd.DataFrame(transactions)
        except Exception as e:
            print(f"Failed to convert transactions to DataFrame: {e}")
            return
    try:
        transactions['date_time'] = pd.to_datetime(transactions['date_time'])
    except Exception as e:
        print(f"Error converting 'date_time' to datetime: {e}")
        return
    # Filter transactions to the last 'n' months
    end_date = datetime.now()  # Use datetime.now() to get a datetime object
    start_date = end_date - timedelta(days=n_months * 30)  # Approximation of months
    filtered_trans = transactions[(transactions['date_time'].dt.date >= start_date.date()) & (transactions['date_time'].dt.date <= end_date.date())]

    # Calculate the variance of the amount spent in each category
    variances = filtered_trans.groupby('category')['amount'].var().sort_values(ascending=False)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    variances.plot(kind='bar', color='skyblue')
    plt.title('Spending Inconsistency by Category')
    plt.xlabel('Category')
    plt.ylabel('Variance of Spending')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Return the category with the highest inconsistency
    most_inconsistent_cat = variances.idxmax()
    inconsistency_value = variances.max()
    return f"The most inconsistent spending category over the last {n_months} months is {most_inconsistent_cat} with a variance of ${inconsistency_value:.2f} in spending."


def analyze_most_spent_category_3_months(transactions, n_months=3):
    # Check and convert transactions to a DataFrame if it's not already one
    if not isinstance(transactions, pd.DataFrame):
        try:
            transactions = pd.DataFrame(transactions)
        except Exception as e:
            print(f"Failed to convert transactions to DataFrame: {e}")
            return

    # Ensure the 'date_time' column is in datetime format
    try:
        transactions['date_time'] = pd.to_datetime(transactions['date_time'])
    except Exception as e:
        print(f"Error converting 'date_time' to datetime: {e}")
        return
    # Define the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_months * 30)  # Approximation of months

    # Filter transactions
    filtered_trans = transactions[
        (transactions['date_time'].dt.date >= start_date.date()) & (transactions['date_time'].dt.date <= end_date.date())]

    # Calculate total spend per category
    total_spend = filtered_trans.groupby('category')['amount'].sum().sort_values(ascending=False)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    total_spend.plot(kind='bar', color='lightgreen')
    plt.title('Total Spend by Category in the Last 3 Months')
    plt.xlabel('Category')
    plt.ylabel('Total Spend')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Return the category with the highest spend
    most_spent_cat = total_spend.idxmax()
    spent_value = total_spend.max()
    return f"The category with the most spent over the last {n_months} months is {most_spent_cat}, with a total of ${spent_value:.2f}."


def visualize_spent_by_category(transactions):
    # Ensure transactions is a DataFrame
    if not isinstance(transactions, pd.DataFrame):
        transactions = pd.DataFrame(transactions)

    # Group the transactions by category and sum the amounts
    category_spend = transactions.groupby('category')['amount'].sum().reset_index()

    # Calculate the total spend
    total_spend = category_spend['amount'].sum()

    # Format the total spend with commas
    total_spend_formatted = f"${total_spend:,.2f}"

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    wedges, texts, autotexts = ax.pie(category_spend['amount'], labels=category_spend['category'],
                                      autopct='%1.1f%%', startangle=140, pctdistance=0.85,
                                      wedgeprops=dict(width=0.4))  # This creates the doughnut shape

    # Customize autopct labels (percentages)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)

    # Set the title and remove the x and y axis labels
    plt.title('Total Spent by Category')

    # Place the total spend in the center
    plt.text(0, 0, f'Total\n{total_spend_formatted}', ha='center', va='center', fontsize=12)

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Return the DataFrame for further use if needed
    return category_spend


def process_query(query, transactions, intent_model, ner_model, tokenizer):
    # Tokenize and pad the query for the intent model
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=50)

    # Predict the intent
    intent_pred = intent_model.predict(padded_sequence)
    predicted_intent_id = np.argmax(intent_pred, axis=1)[0]
    predicted_intent = intent_encoder.inverse_transform([predicted_intent_id])[0]

    # Use the NER model to extract entities
    ner_pred = ner_model.predict(padded_sequence)
    predicted_ner_ids = np.argmax(ner_pred, axis=2)[0]  # Adjust for TensorFlow
    predicted_ner_labels = [label_tokenizer.index_word.get(id, 'O') for id in predicted_ner_ids]

    # Extract entities (you'll need to adjust this based on how your NER labels are structured)
    entities = {'CATEGORY': None, 'TIME_FRAME': None}
    for word, label in zip(query.split(), predicted_ner_labels):
        if label.startswith('B-CATEGORY') or label.startswith('I-CATEGORY'):
            entities['CATEGORY'] = word
        elif label.startswith('B-TIME_FRAME') or label.startswith('I-TIME_FRAME'):
            entities['TIME_FRAME'] = word

    # Call the appropriate analysis function based on the predicted intent
    if predicted_intent == 'total_spend_category':
        # Assuming that 'CATEGORY' entity was extracted
        if 'CATEGORY' in entities and entities['CATEGORY'] is not None:
            category = entities['CATEGORY']
            return visualize_spent_by_category(transactions[transactions['category'].str.contains(category, case=False)])
        else:
            # If the category was not specified, show for all categories
            return visualize_spent_by_category(transactions)
    elif predicted_intent == 'most_inconsistent_category':
        return analyze_spending_inconsistency(transactions)
    elif predicted_intent == 'most_spent_category_3_months':
        return analyze_most_spent_category_3_months(transactions)
    # Add more elif blocks for other intents here if necessary
    else:
        return "Sorry, I didn't understand your query. Please try again."

def interactive_session():
    print("Welcome to the Transaction Analysis System! Type 'exit' to leave.")
    while True:
        # Get user input
        user_query = input("\nPlease enter your query: ").strip()
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        # Process the query and print the response
        response = process_query(user_query, synthetic_transactions, intent_model, ner_model, tokenizer)
        print(response)

# Run the interactive session
interactive_session()
