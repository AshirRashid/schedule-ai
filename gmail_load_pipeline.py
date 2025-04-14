import os
import json
import base64
from bs4 import BeautifulSoup
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from langchain.schema import Document
from langchain.chains import TransformChain, LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def fetch_gmail_documents(n=10):
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_info(
            json.loads(open('token.json').read()))
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'gcal_main_new_secret.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    results = service.users().messages().list(
        userId='me', maxResults=n, labelIds=['INBOX']).execute()
    messages = results.get('messages', [])

    documents = []

    for message in messages:
        msg = service.users().messages().get(
            userId='me', id=message['id'], format='full').execute()
        headers = msg['payload']['headers']
        subject = next(
            (h['value'] for h in headers if h['name'].lower() == 'subject'), 'No Subject')
        sender = next(
            (h['value'] for h in headers if h['name'].lower() == 'from'), 'Unknown')
        date = next((h['value']
                    for h in headers if h['name'].lower() == 'date'), 'Unknown')

        body = ""
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body'].get(
                        'data', '').encode('ASCII')).decode('utf-8')
                    break
        elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
            body = base64.urlsafe_b64decode(
                msg['payload']['body']['data'].encode('ASCII')).decode('utf-8')

        documents.append(Document(page_content=body, metadata={
            "id": message['id'],
            "from": sender,
            "subject": subject,
            "date": date
        }))

    print(f"Fetched {len(documents)} emails")
    return {"documents": documents}


def clean_email_docs(inputs):
    documents = inputs["documents"]
    llm = ChatOllama(model="llama3", temperature=0)
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Clean the following email. Only keep the body. Remove all HTML and CSS.\n\n{text}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    cleaned_documents = []
    for doc in documents:
        response = chain.invoke(doc.page_content)
        cleaned_documents.append(
            Document(page_content=response['text'], metadata=doc.metadata))

    print(f"Cleaned {len(cleaned_documents)} emails with LLM")
    return {"cleaned_documents": cleaned_documents}


cleaner_chain = TransformChain(
    input_variables=["documents"],
    output_variables=["cleaned_documents"],
    transform=clean_email_docs
)


def chunk_documents(inputs):
    documents = inputs["cleaned_documents"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len
    )

    chunks = []
    for doc in documents:
        content = f"From: {doc.metadata['from']}\nSubject: {doc.metadata['subject']}\nDate: {doc.metadata['date']}\n\n{doc.page_content}"
        splits = text_splitter.split_text(content)
        chunks.extend(splits)

    print(f"Created {len(chunks)} chunks")
    return {"chunks": chunks}


chunker_chain = TransformChain(
    input_variables=["cleaned_documents"],
    output_variables=["chunks"],
    transform=chunk_documents
)


def save_chunks_to_chroma(inputs):
    chunks = inputs["chunks"]
    embedding_function = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
    chroma_db = Chroma.from_texts(
        chunks, embedding_function, persist_directory="chroma_db")
    chroma_db.persist()

    print(f"Saved {len(chunks)} chunks to ChromaDB")
    return {"status": "done", "chunks_saved": len(chunks)}


save_chain = TransformChain(
    input_variables=["chunks"],
    output_variables=["status", "chunks_saved"],
    transform=save_chunks_to_chroma
)


pipeline = SequentialChain(
    chains=[cleaner_chain, chunker_chain, save_chain],
    input_variables=["documents"],
    output_variables=["status", "chunks_saved"],
    verbose=True
)


if __name__ == "__main__":
    initial_data = fetch_gmail_documents(n=3)
    result = pipeline.invoke(initial_data)
    print(f"Pipeline complete: {result}")
    breakpoint()
