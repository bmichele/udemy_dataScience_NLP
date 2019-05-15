# Sentiment analyzer

## What it does
* find out sentiment related to text
* useful to classify reviews

## What we'll do
* just consider review for electronics
* the same code can be used on other data
* we'll use xml obtained with bs

Two passes needed
* transform textual data into numerical data
* fit sklearn model

The obtained code can be used on other numerical input

## Tokenization

In python we can use the split function, e.g. `'sono una stringa'.split()`
But split does not allow to deal with punctuation. Better to use nltk.word_tokenize(). It is just a fancy string split...

Custom tokenizer can be used for special purpose, e.g. if I want to remove stopwords, numbers, symbols, ...

## Tokens to vectors
We will try here to write code to get feature vectors without using sklearn, in order to think about how it can be done.
### Binary Bag Of Words (BOW)
In the vector, the entry related to the word is 0 if the word is not present, or 1 in case it appears in the document.
Good to start with a `np.zeros` vector, then switch to 1 the elements related to words that are present.
The switch can be controlled using a dictionary word_to_index

### Counts
As Binary BOW, but count the number of occurrencies instead of putting 1.
problem with counting is that long sentences will have long BOWs.
This can happen in sentiment analysis, as some sentences are short and some are really long.
We do not want to differentiate the documents based on length of the vector, so we can normalize it.
Normalization is also important as in a Neural Network we might have the sigmoid adctivation function, and the sigmoid approaches 1 very quickly (so that s(10) is approximately s(100) and close to 1).
