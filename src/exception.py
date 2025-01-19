import sys # Provides access to system-specific parameters and functions (e.g., exception handling)
# from src.logger import logging # For tracking events during runtime

# Utility function to create a detailed error message
def error_message_detail(error, error_detail: sys):
    """
    Generates a detailed error message, including:
    - The filename where the error occurred.
    - The line number of the error.
    - The actual error message.

    Parameters:
    error (Exception): The caught exception object.
    error_detail (sys): sys module to access exception traceback.

    Returns:
    str: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extract traceback details (type, value, and traceback object)
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the name of the script where the exception occurred
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)  # Format the error details: filename, line number, and error message
    )
    return error_message  # Return the formatted error message
    
# Custom exception class for detailed error reporting
class CustomException(Exception):
    """
    A custom exception class to provide detailed error messages,
    including the filename, line number, and error message.

    Inherits from Python's built-in Exception class.
    """

    def __init__(self, error_message, error_detail: sys):
        """
        Constructor for the CustomException class.

        Parameters:
        error_message (str): A custom message describing the error.
        error_detail (sys): sys module to access exception traceback details.
        """
        super().__init__(error_message)  # Initialize the parent Exception class with the error message
        # Use the error_message_detail function to generate a detailed error message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """
        Override the string representation of the exception to return the detailed error message.

        Returns:
        str: The detailed error message.
        """
        return self.error_message  # Return the detailed error message when the exception is converted to a string
    
# if __name__ == "__main__":
#     try:
#         a=1/0
#     except Exception as e:
#         logging.info("Divide by Zero")
#         raise CustomException(e, sys)