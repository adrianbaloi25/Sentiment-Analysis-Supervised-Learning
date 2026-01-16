import re
import string

def clean_text(text):
    """
    Funcție pentru curățarea textului:
    1. Transformă în minuscule.
    2. Elimină semnele de punctuație.
    3. Elimină spațiile multiple.
    """
    if not isinstance(text, str):
        return ""
    
    # Transformare în minuscule
    text = text.lower()
    
    # Eliminare tag-uri HTML (dacă există)
    text = re.sub(r'<.*?>', '', text)
    
    # Eliminare punctuație
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Eliminare spații albe extra
    text = text.strip()
    
    return text

def prepare_features(df, text_column):
    """
    Aplică curățarea pe o întreagă coloană dintr-un DataFrame.
    """
    return df[text_column].apply(clean_text)