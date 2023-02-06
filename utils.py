import string

def clean_description(desc):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    # # tokenize
    # desc = desc.split()
    # convert to lower case
    desc = [word.lower() for word in desc]
    # remove punctuation from each token
    desc = [w.translate(table) for w in desc]
    #remove numbers
    table = str.maketrans('', '', string.digits)
    desc = [w.translate(table) for w in desc]
    # remove one letter words except 'a'
    desc = [word for word in desc if len(word)>1 or word == 'a']


    return desc

