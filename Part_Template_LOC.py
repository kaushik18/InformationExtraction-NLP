# SpaCy Imports
import neuralcoref    
import spacy

# Initialization of the Spacy NLP Pipeline 
nlp = spacy.load('en_core_web_sm')

# Adding the Neural Coreference Resolution to the Pipeline
neuralcoref.add_to_pipe(nlp)

# Merge Entities for the re-tokenization of the function
def merge_entities(document):
    with document.retokenize() as retokenizer:
        for entity in document.ents:
            retokenizer.merge(entity)
    return document

# Location Extraction Function
def location_pattern(text):

    # Pass the sentence through the Pipeline
    doc = nlp(text)

    # Re-Tokenize the Document
    document = merge_entities(doc)
    
    # Initialization of the selected Text for Locations
    text_selection = []
    
    # Position Initialization
    index_next_char = -1
  
    # Loop through the Entities
    for entity in document.ents:

        # Check for Location
        if(entity.label_ == "GPE"):

            # First Location Case and check for Start Character in the Loop
            if(index_next_char == -1 or index_next_char == entity.start_char):
                text_selection.append(entity.text)

                # Jump 2 ahead to go to next word (Consists of word,entity,word,entity ordering)
                index_next_char = entity.end_char + 2

            # If not the first location or not the first character and check for 'and' type single locations
            elif(index_next_char > 0 and index_next_char + 3 < len(text) and (index_next_char + 3 == entity.start_char or index_next_char + 4 == entity.start_char) and (text[index_next_char - 1 : index_next_char + 2] == "and" or text[index_next_char : index_next_char + 3] == "and")):
                text_selection.append("&")
                text_selection.append(entity.text)
                index_next_char = entity.end_char + 2
            
            # Comma Separted Different Locations
            else:
                text_selection.append("#")
                text_selection.append(entity.text)
                index_next_char = entity.end_char + 2
  
    # To find the Remaining Location
    remaining_list = []
  
    # Loop thorugh the selected Texts
    for i in range(len(text_selection)):

        # 'and' Type Single Locations
        if(text_selection[i] == "&"):
            # Adding the two plus words in the location
            remaining_list.append(i)
            remaining_list.append(i+1)
            j = i - 1

            # Two Separate Locations
            while(j != 0 and text_selection[j+1] != "#"):
                # appending them one by one but separately to identify the different locations
                remaining_list.append(j)
                j = j - 1
  
    # Creation of the list from the location dictionaries
    remaining_list = list(dict.fromkeys(remaining_list))
  
    for index in sorted(remaining_list, reverse=True):
        del text_selection[index]
  
    text_pattern = []
    temp_tuple_pattern = ()
    
    # Final Location patterns, Looking for single multiple word locations or multiple locations
    for i in (range(len(text_selection)-1)):
        if(text_selection[i+1] == "#" or text_selection[i] == "#"):
            continue
        else:
            temp_tuple_pattern = (text_selection[i],text_selection[i+1])
            text_pattern.append(temp_tuple_pattern)
  
    return text_pattern
        
# Sentence Filter
def part_template_sentence_check(sentence,ner_sentence):

    count = sum((x == 'GPE') for x in ner_sentence.values())
    if(count>=2):
        return sentence
    
    return None

# Main Part(Location,Location) Call
def part_home(sentence):
    
    # Different Approaches and Selection of Unique Sentences
    combined_output=set()

    # Call for Part Template Extraction
    combined_output.update(location_pattern(sentence))

    # Conversion of Set to List
    combined_output = list(combined_output)
    
    # Final Output of sentences and Templates
    return combined_output

def getPart(sentences,ner_list):

    # Initialization of the final list of templates
    final_part = []

    # Filtered Sentence List
    selected_sentences_list = []
    
    # Looping through Sentneces One by one to Filter
    for index,sentence in enumerate(sentences):
        output_sentence = part_template_sentence_check(sentence,ner_list[index])
        if output_sentence is None:
            continue
        else:
            selected_sentences_list.append(output_sentence)
    
    # Looping through Selected Sentences for Template Extraction 
    for sentence in selected_sentences_list:
        try:
            # Part(Location,Location) Template Extraction Call
            part_output = part_home(sentence)

            # If no Template then Continue
            if(part_output != []):
                for j in part_output:

                    # Formatting of the Part(Location,Location) Template
                    temp_dict ={}
                    temp_dict["template"] ="PART_OF"
                    temp_dict["sentences"] = []
                    temp_dict["sentences"].append(sentence)
                    temp_dict["arguments"] = {}
                    temp_dict["arguments"]["1"] = j[0]
                    temp_dict["arguments"]["2"] = j[1]
                    
                    # Final Template appended to list to be returned to the calling function
                    final_part.append(temp_dict)
        
        except:
            continue    

    return final_part