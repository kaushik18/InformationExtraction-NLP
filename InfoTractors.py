# NLP - Information Extraction Templates

# Basic Imports
import sys
import glob

from Features import Features_Extraction
from Templates import Template_Extraction

# ------------------------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------------------ Task 3 - Main Program---Driver Code ----------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------ #

if(len(sys.argv) < 2):

    files_list = glob.glob('WikipediaArticles\*.txt')
    for file_name in files_list:

        # ------------------------------------------------------------------------------------------------------------------------------------ #
        # ------------------------------------------- Task 1 - NLP Features from Input Text File --------------------------------------------- #
        # ------------------------------------------------------------------------------------------------------------------------------------ #
        sentences,words_list,pos_tag_list,wordnet_tagged_list,stemmas_list,lemmas_list,lemmas_wordnet_list,synonymns_list,hypernyms_list,hyponyms_list,meronyms_list,holonyms_list,dependency_parse_tree_list,ners_list = Features_Extraction(file_name)
        
        # ------------------------------------------------------------------------------------------------------------------------------------ #
        # ------------------- Task 2 - Extract Information Templates using Heuristic, or Statistical or Both Methods ------------------------- #
        # ------------------------------------------------------------------------------------------------------------------------------------ #
        Template_Extraction(sentences,dependency_parse_tree_list,ners_list,file_name)

else:

    file_name = sys.argv[1]
    
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # ------------------------------------------- Task 1 - NLP Features from Input Text File --------------------------------------------- #
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    sentences,words_list,pos_tag_list,wordnet_tagged_list,stemmas_list,lemmas_list,lemmas_wordnet_list,synonymns_list,hypernyms_list,hyponyms_list,meronyms_list,holonyms_list,dependency_parse_tree_list,ners_list = Features_Extraction(file_name)
        
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    # ------------------- Task 2 - Extract Information Templates using Heuristic, or Statistical or Both Methods ------------------------- #
    # ------------------------------------------------------------------------------------------------------------------------------------ #
    Template_Extraction(sentences,dependency_parse_tree_list,ners_list,file_name)


print('\nTemplate Extraction Completed\n')