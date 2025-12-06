
import sys
import importlib.util

def check_install(package_name):
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def extract_text(pdf_path):
    text = ""
    try:
        # Try PyPDF2/pypdf first
        if check_install('PyPDF2'):
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        elif check_install('pypdf'):
             import pypdf
             with open(pdf_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        else:
            print("Error: neither PyPDF2 nor pypdf is installed.")
            return None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
    return text

if __name__ == "__main__":
    path = r"C:\Users\dygks\Documents\Dev\team_project\Breaking the trend.pdf"
    content = extract_text(path)
    if content:
        # Print first 5000 chars to avoid overwhelming buffer, or write to file
        with open("pdf_content.txt", "w", encoding="utf-8") as f:
            f.write(content)
        print("Successfully extracted text to pdf_content.txt")
        print("First 2000 characters:")
        print(content[:2000])
    else:
        print("Failed to extract content.")
