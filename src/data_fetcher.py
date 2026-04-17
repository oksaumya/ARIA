import fitz
from docx import Document
import wikipedia

def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_bytes = file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
            for page in pdf:
                extracted = page.get_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        pass
    return text

def extract_text_from_docx(file):
    text = ""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        pass
    return text

def extract_text_from_txt(file):
    try:
        return file.getvalue().decode("utf-8")
    except Exception:
        return ""

def fetch_wikipedia_content(query):
    try:

        page = wikipedia.page(query, auto_suggest=True)
        return page.content
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            return wikipedia.page(e.options[0], auto_suggest=False).content
        except:
            return ""
    except wikipedia.exceptions.PageError:
        return ""
    except Exception as e:
        return ""
