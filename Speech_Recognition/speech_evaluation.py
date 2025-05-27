"""
This script calculates differences and error percentages between two text files.
It takes two file paths as command-line arguments, reads the text content from these files,
identifies uncommon words, and computes an error percentage based on these uncommon words
relative to the total number of words in each file.
"""
import argparse
import sys

def UncommonWords(A, B): 
    """
    Identifies words that appear in only one of the two input strings.

    This function counts word occurrences in both strings. A word is considered
    "uncommon" if its total count across both strings is 1, meaning it's
    unique to either string A or string B.

    Args:
        A (str): The first input string.
        B (str): The second input string.

    Returns:
        list: A list of words that are uncommon to string A or string B.
    """
	# count will contain all the word counts 
	count = {} 
	
	# insert words of string A to hash 
	for word in A.split(): 
		count[word] = count.get(word, 0) + 1
	
	# insert words of string B to hash 
	for word in B.split(): 
		count[word] = count.get(word, 0) + 1

	# return required list of words 
	return [word for word in count if count[word] == 1] 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate differences and error percentages between two text files.')
    parser.add_argument('fileA_path', type=str, help='Path to the first text file (A).')
    parser.add_argument('fileB_path', type=str, help='Path to the second text file (B).')
    args = parser.parse_args()

    fileA_path_arg = args.fileA_path
    fileB_path_arg = args.fileB_path
    
    A_text = ""
    B_text = ""

    try:
        with open(fileA_path_arg, "r") as f1:
            A_text = f1.read()
    except FileNotFoundError:
        print(f"Error: File A not found at {fileA_path_arg}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file A ({fileA_path_arg}): {e}")
        sys.exit(1)
        
    try:
        with open(fileB_path_arg, "r") as f2:
            B_text = f2.read()
    except FileNotFoundError:
        print(f"Error: File B not found at {fileB_path_arg}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file B ({fileB_path_arg}): {e}")
        sys.exit(1)

    # Print required answer 
    uncommon_words_list = UncommonWords(A_text, B_text)
    print("Uncommon words:", uncommon_words_list)

    # Calculate half the number of uncommon words. This value is used as a basis for an error metric.
    # Each uncommon word contributes to a difference; dividing by 2 normalizes this somewhat
    # if considering pairs of differences (one word in A not in B, one word in B not in A).
    half_num_uncommon_words = len(uncommon_words_list) / 2.0
    #print(half_num_uncommon_words)

    num_words_A = len(A_text.split()) # Simplified word counting
    num_words_B = len(B_text.split()) # Simplified word counting
    
    print("Number of words in A:", num_words_A)
    print("Number of words in B:", num_words_B)

    # The commented-out code block for 'c' (difference in word counts) has been removed.
    
    # Calculate error percentage with respect to file A
    if num_words_A == 0:
        print("percentage error with respect to A: N/A (File A is empty)")
        error_percentage_A = "N/A"
    else:
        # This percentage represents the proportion of "errors" (based on half_num_uncommon_words)
        # relative to the total number of words in file A.
        error_percentage_A = (half_num_uncommon_words * 100) / num_words_A
        print("percentage error with respect to A:", error_percentage_A)
        
    # Calculate error percentage with respect to file B
    if num_words_B == 0:
        print("percentage error with respect to B: N/A (File B is empty)")
        error_percentage_B = "N/A"
    else:
        # This percentage represents the proportion of "errors" (based on half_num_uncommon_words)
        # relative to the total number of words in file B.
        error_percentage_B = (half_num_uncommon_words * 100) / num_words_B
        print("percentage error with respect to B:", error_percentage_B)

