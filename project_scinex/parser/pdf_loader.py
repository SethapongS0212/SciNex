import fitz

def load_pdf(path):
    return fitz.open(path)