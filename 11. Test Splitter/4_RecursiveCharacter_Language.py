from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

file_path = "./file-example_PDF_1MB.pdf"

loader = PyPDFLoader(file_path)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
    )

result = splitter.split_documents(documents)
print(f"Total chunks: {len(result)}")
print(result[0].page_content)"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

result = splitter.split_text(text)
print(f"Total chunks: {len(result)}")
for i, chunk in enumerate(result):
    print(f"--- Chunk {i+1} ---")
    print(chunk)    