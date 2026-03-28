import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    try:
        with zipfile.ZipFile(docx_path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            
            # The XML namespaces used in docx files
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            
            text = []
            for paragraph in tree.iterfind('.//w:p', namespaces):
                para_text = []
                for run in paragraph.iterfind('.//w:r', namespaces):
                    for text_node in run.iterfind('.//w:t', namespaces):
                        if text_node.text:
                            para_text.append(text_node.text)
                if para_text:
                    text.append(''.join(para_text))
            
            return '\n'.join(text)
    except Exception as e:
        return f"Error reading {docx_path}: {e}"

if __name__ == '__main__':
    if len(sys.argv) > 1:
        text = extract_text_from_docx(sys.argv[1])
        with open('output_utf8.txt', 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        print("Please provide docx path")
