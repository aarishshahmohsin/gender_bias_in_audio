import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

def generate_sentence(word):
    doc = nlp(word)
    for token in doc:
        if token.pos_ == "NOUN":
            return f"A {word} taking"
        else:
            return f"A {word} person talking"

# Example usage
print(generate_sentence("dog"))   # Output: "A dog taking"
print(generate_sentence("running")) # Output: "A running person talking"
