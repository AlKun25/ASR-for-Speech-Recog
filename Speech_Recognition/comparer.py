"""
Calculates the cosine similarity between a given input query string and
the content of a specified text file.
The script takes the input query and the database file path as command-line arguments.
It then prints the match percentage.
"""
import re
import math
import argparse


def _preprocess_text(text_input):
    """
    Preprocesses a text string by converting to lowercase, removing punctuation,
    and splitting into a list of words.

    Args:
        text_input (str): The text string to preprocess.

    Returns:
        list: A list of words from the processed text.
    """
    lowercase_text = text_input.lower()
    # Replace punctuation with space, then split into words
    word_list = re.sub(r"[^\w\s]", "", lowercase_text).split() # Keep spaces for split, remove other punctuation
    return [word for word in word_list if word] # Remove empty strings that might result from multiple spaces

def _calculate_tf(word_list, universal_set_of_words):
    """
    Calculates the Term Frequency (TF) vector for a given word list
    based on a universal set of words.

    Args:
        word_list (list): A list of words from a document.
        universal_set_of_words (list): A list of all unique words across all documents.

    Returns:
        list: A TF vector where each element is the frequency of a word
              from the universal set in the given word_list.
    """
    tf_vector = []
    for universal_word in universal_set_of_words:
        tf_counter = 0
        for word_in_doc in word_list:
            if universal_word == word_in_doc:
                tf_counter += 1
        tf_vector.append(tf_counter)
    return tf_vector

def cosineSimilarity(inputQuery_text, databaseFilePath):
    """
    Calculates and prints the cosine similarity between an input query string
    and the text content of a database file.

    The function performs the following steps:
    1. Preprocesses the input query and the database text (lowercase, remove punctuation, split into words).
    2. Creates a universal set of unique words from both the query and the database.
    3. Calculates Term Frequency (TF) vectors for both the query and the database.
    4. Computes the dot product of the TF vectors.
    5. Computes the magnitudes of each TF vector.
    6. Calculates the cosine similarity and prints it as a percentage.

    Args:
        inputQuery_text (str): The raw input query string.
        databaseFilePath (str): The file path to the database text file.
    
    Prints:
        str: The match percentage or an error message if issues occur (e.g., file not found, division by zero).
    """
	
    universalSetOfUniqueWords = []
    matchPercentage = 0

    # Section 1: Text preprocessing for query and database
    # Preprocess input query
    queryWordList = _preprocess_text(inputQuery_text)

    # Read and preprocess database text
    try:
        with open(databaseFilePath, "r") as fd:
            database_text = fd.read() # Renamed from database1
    except FileNotFoundError:
        print(f"Error: Database file not found at {databaseFilePath}")
        return
    except Exception as e:
        print(f"Error reading database file: {e}")
        return
    databaseWordList = _preprocess_text(database_text)

    # Section 2: Building the universal set of words
    # Populate universalSetOfUniqueWords from query
    for word in queryWordList:
        if word not in universalSetOfUniqueWords:
            universalSetOfUniqueWords.append(word)
    # Populate universalSetOfUniqueWords from database
    for word in databaseWordList:
        if word not in universalSetOfUniqueWords:
            universalSetOfUniqueWords.append(word)
    
    if not universalSetOfUniqueWords:
        print("Error: No words found in query or database after preprocessing.")
        return

    # Section 3: Calculating Term Frequency (TF) vectors
    queryTF = _calculate_tf(queryWordList, universalSetOfUniqueWords)
    databaseTF = _calculate_tf(databaseWordList, universalSetOfUniqueWords)

    # Section 4: Calculating dot product
    dotProduct = 0
    for i in range (len(queryTF)):
        dotProduct += queryTF[i]*databaseTF[i]

    # Section 5: Calculating vector magnitudes
    queryVectorMagnitude = 0
    for i in range (len(queryTF)):
        queryVectorMagnitude += queryTF[i]**2
    queryVectorMagnitude = math.sqrt(queryVectorMagnitude)

    databaseVectorMagnitude = 0
    for i in range (len(databaseTF)):
        databaseVectorMagnitude += databaseTF[i]**2
    databaseVectorMagnitude = math.sqrt(databaseVectorMagnitude)

    # Section 6: Calculating cosine similarity
    if queryVectorMagnitude == 0 or databaseVectorMagnitude == 0:
        print("Error: Cannot calculate similarity because one or both texts have no recognized words after processing (zero vector magnitude).")
        matchPercentage = 0.0
    else:
        matchPercentage = (float)(dotProduct / (queryVectorMagnitude * databaseVectorMagnitude))*100

	'''
	print queryWordList
	print
	print databaseWordList


	print queryTF
	print
	print databaseTF
	'''

	output = "Input query text matches %0.02f%% with database."%matchPercentage

	print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Calculate cosine similarity between an input query string and the content of a text file.'
    )
    parser.add_argument('inputQuery', type=str, help='The input query string.')
    parser.add_argument('databaseFile', type=str, help='Path to the database text file.')
    args = parser.parse_args()

    # Use parsed arguments directly
    cosineSimilarity(args.inputQuery, args.databaseFile)
