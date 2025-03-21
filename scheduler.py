import schedule
import time

def task():
    print("Running scheduled task...")
    # Your task implementation

# Schedule the task to run every day at a specific time
schedule.every().day.at("07:30").do(task)

while True:
    schedule.run_pending()
    time.sleep(1)
