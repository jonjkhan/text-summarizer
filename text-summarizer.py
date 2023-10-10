import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download("punkt")
nltk.download("stopwords")


def summarize_text(text, num_sentences=3):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text)

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
 
    # Calculate word frequency
    word_freq = FreqDist(words)

    # Assign a score to each sentence based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word]
                else:
                    sentence_scores[sentence] += word_freq[word]

    # Get the top N sentences with the highest scores
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]

    # Join the selected sentences to create the summary
    summary = TreebankWordDetokenizer().detokenize(summary_sentences)

    return summary


# Example usage
if __name__ == "__main__":
    input_text = """
    My name is Jonathan Khan and I am a second-year Computer
                         Science student with a passion for coding. I am Proficient
                          in Python and experienced with tools like VSCode,
                           Wing, and PyCharm, I also have some exposure into
                            the world of web development with HTML and CSS, along with some
                            exposure to Git and Netlify.
                             I'm actively seeking an internship opportunity
                              this summer to further expand my skills and contribute
                               to exciting projects at your company
    """

    summary = summarize_text(input_text)
    print(summary)
