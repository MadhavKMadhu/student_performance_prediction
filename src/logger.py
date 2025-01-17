import logging # For logging messages to a file or console
import os # For interacting with the operating system (e.g., file paths, directories)
from datetime import datetime # For working with dates and times

# Generate a unique log file name based on the current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the path for the logs directory and include the log file
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the logs directory if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Complete path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logging system
logging.basicConfig(
    filename = LOG_FILE_PATH, # Set the file to which logs will be written
    format = "[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s", # Define the log message format
    level = logging.INFO, # Set the logging level to INFO (captures INFO, WARNING, ERROR, CRITICAL)
)

# # Main block: This ensures the following code runs only when the script is executed directly
# if __name__ == "__main__":
#     # Log an informational message to indicate that logging has started
#     logging.info("Logging has started")