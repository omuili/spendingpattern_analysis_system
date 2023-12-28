import random
import csv
from datetime import datetime, timedelta
from faker import Faker
import uuid

# Initialize Faker
fake = Faker()

# Constants
categories = {
    "Groceries": 101, "Electronics": 102, "Dining": 103, "Utilities": 104, "Entertainment": 105
}
MERCHANTS = [
    "Whole Foods", "Walmart", "Trader Joe's", "Best Buy", "Apple Store",
    "Newegg", "McDonald's", "Starbucks", "Chipotle", "City Water", "Energy Power",
    "InternetCo", "Netflix", "AMC", "Spotify"
]
DEVICES = ["Mobile", "Desktop", "In-Store Terminal"]
STATUS = "Cleared"
TRANSACTION_TYPE = "Credit Card"


# Function to generate a random date and time within the last 24 months
def random_datetime():
    days_ago = random.randint(1, 730)
    random_datetime = datetime.now() - timedelta(days=days_ago)
    return random_datetime.strftime("%Y-%m-%d %H:%M:%S")


# Function to generate a random merchant, category, and category code
def random_merchant_category():
    merchant = random.choice(MERCHANTS)
    category, code = random.choice(list(categories.items()))
    return merchant, category, code


# Function to generate a realistic transaction amount as a float
def random_amount(min=5, max=500):
    return round(random.uniform(min, max), 2)


# Function to generate a random user ID and account number
def random_user_and_account():
    user_id = fake.random_int(min=1000, max=9999)
    account_num = fake.bothify(text='####-####-####-####')  # Random account number
    return user_id, account_num


# Function to calculate tax amount (assuming a flat 10% tax rate for simplicity)
def calculate_tax(amount):
    return round(amount * 0.10, 2)


# Generate Transactions
def generate_transactions(num_transactions=1000):
    transactions = []
    user_balances = {}  # Track user balances
    for _ in range(num_transactions):
        user_id, account_num = random_user_and_account()
        datetime = random_datetime()
        merchant, category, category_code = random_merchant_category()
        amount = random_amount()
        tax_amount = calculate_tax(amount)
        device = random.choice(DEVICES)
        transaction_id = str(uuid.uuid4())
        description = f"{category} purchase at {merchant}"

        # Calculate running balance for each user
        prev_balance = user_balances.get(user_id, 1000)  # Starting balance of 1000 for each user
        new_balance = prev_balance - amount - tax_amount

        transactions.append({
            "transaction_id": transaction_id,
            "date_time": datetime,
            "user_id": user_id,
            "account_num": account_num,
            "type": TRANSACTION_TYPE,
            "status": STATUS,
            "merchant": merchant,
            "description": description,
            "amount": amount,  # Store as a number
            "tax_amount": tax_amount,  # Store as a number
            "category": category,
            "category_code": category_code,
            "device_used": device,
            "previous_balance": prev_balance,  # Store as a number
            "new_balance": new_balance  # Store as a number
        })

        # Update user balance
        user_balances[user_id] = new_balance

    return transactions


# Save transactions to a CSV file
def save_to_csv(transactions, filename='transactions.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=transactions[0].keys())
        writer.writeheader()
        writer.writerows(transactions)


# Main function to generate and save transactions
def main():
    num_transactions = 1000  # Number of transactions to generate
    transactions = generate_transactions(num_transactions)
    save_to_csv(transactions)  # Save the transactions to a CSV file


# Run the script
if __name__ == "__main__":
    main()
