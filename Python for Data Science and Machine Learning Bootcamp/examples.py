import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# NUMPY
##### Grab the email website domain from a string
def domainGet(email):
    return email.split('@')[1]


domainGet('user@domain.com')


##### Returns True if the word 'dog' is contained in the input string
def findDog(st):
    return 'dog' in st.lower().split()


findDog('Is there a dog here?')


##### Count the number of times the word "dog" occurs in a string
def countDog(st):
    count = 0
    for word in st.lower().split():
        if word == 'dog':
            count += 1
    return count


countDog('This dog runs faster than the other dog dude!')


##### Filter out words from a list that don't start with the letter 's'
seq = ['soup', 'dog', 'salad', 'cat', 'great']
list(filter(lambda word: word[0] == 's', seq))


##### Return one of 3 possible results. If your speed is 60 or less, the result is "No Ticket". If speed is between 61 and 80 inclusive, the result is "Small Ticket". If speed is 81 or more, the result is "Big Ticket". Unless it is your birthday (encoded as a boolean value in the parameters of the function) -- on your birthday, your speed can be 5 higher in all cases
def caught_speeding(speed, is_birthday):
    if is_birthday:
        speeding = speed - 5
    else:
        speeding = speed

    if speeding > 80:
        return 'Big Ticket'
    elif speeding > 60:
        return 'Small Ticket'
    else:
        return 'No Ticket'


caught_speeding(81, True)
caught_speeding(81, False)


# PANDAS
##### Find who has the word Chief in their complex job title
def chief_string(title):
    if 'chief' in title.lower():
        return True
    else:
        return False


##### How many people have a credit card expiring in 2025?
sum(dataframe['CC Exp Date'].apply(lambda x: x[3:]) == '25')


##### Top 5 most popular email providers
dataframe['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(5)
