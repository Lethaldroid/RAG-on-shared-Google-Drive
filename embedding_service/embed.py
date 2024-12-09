__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import gspread
from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import os
import pickle
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
import argparse
import shutil
import sys
import time
import socket
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from datetime import datetime, timedelta


# Define the scopes needed for accessing Google Drive and Sheets
SCOPES = [
    'https://www.googleapis.com/auth/drive',
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/documents',
    'https://www.googleapis.com/auth/presentations'
]

# Authenticate and create a credentials object
def authenticate_google_drive():
    creds = None
    # Check if token.pickle exists (stores user credentials after initial login)
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    # If there are no valid credentials, prompt user to log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
            #  creds = Credentials.from_service_account_file('service_account.json')

        # Save the credentials for future use
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds

# Authenticate and initialize gspread client
creds = authenticate_google_drive()
gc = gspread.authorize(creds)


def get_embedding_function():
    # Initialize HuggingFace embeddings with the LaBSE model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
    return embeddings

# Constants for paths
CHROMA_PATH = "chroma"  # Or your desired path


# Connect to Chroma DB (replace with your actual host/port if needed)
client = chromadb.HttpClient(host="localhost", port=8000) 

# Initialize Google Drive and Sheets services using authenticated credentials
drive_service = build('drive', 'v3', credentials=creds)
sheets_service = build('sheets', 'v4', credentials=creds)

shared_drive_id = '0AFNbEiNe12MnUk9PVA'  # Replace with your Shared Drive ID

def main():
    # Filter out unwanted arguments added by Jupyter/IPython.
    filtered_args = [arg for arg in sys.argv if arg.startswith("--reset")]

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args(filtered_args)  # Use the filtered arguments
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_google_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def get_folder_id(folder_name, shared_drive_id):
    """Retrieves the ID of a folder within a Shared Drive."""
    try:
        results = drive_service.files().list(
            q=f"name = '{folder_name}' and mimeType='application/vnd.google-apps.folder' "
              f"and '{shared_drive_id}' in parents",
            corpora='drive',
            driveId=shared_drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="nextPageToken, files(id, name, mimeType, parents)",
            pageSize=1000
        ).execute()
        items = results.get('files', [])
        if items:
            return items[0]['id']
        else:
            return None
    except Exception as e:
        print(f"Error getting folder ID: {e}")
        return None

def calculate_chunk_ids(chunks):
  """Generates unique chunk IDs for each document chunk."""
  last_page_id = None
  current_chunk_index = 0
  for chunk in chunks:
      source = chunk.metadata.get("source")
      page = chunk.metadata.get("page", 1)
      current_page_id = f"{source}:{page}"
      if current_page_id == last_page_id:
          current_chunk_index += 1
      else:
          current_chunk_index = 0
      chunk.metadata["id"] = f"{current_page_id}:{current_chunk_index}"
      last_page_id = current_page_id
  return chunks


def list_all_files_recursive(folder_id, documents):
    """
    Recursively lists all files within a folder and its subfolders,
    filtering for specific MIME types. Fetches content and continues
    listing in parallel.
    """
    page_token = None
    target_mime_types = [
        'application/vnd.google-apps.spreadsheet',
        'application/vnd.google-apps.document',
        'application/vnd.google-apps.presentation'
    ]

    while True:
        results = drive_service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            corpora='drive',
            driveId=shared_drive_id,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="nextPageToken, files(id, name, mimeType, parents)",
            pageSize=1000,
            pageToken=page_token
        ).execute()
        items = results.get('files', [])

        if not items:
            print('No Google Sheets, Google Docs, or Google Slides found.')
        else:
            # Filter for desired MIME types and fetch content in parallel
            filtered_items = [item for item in items if item['mimeType'] in target_mime_types]

            # Start fetching content for this batch of files
            for item in filtered_items:
                print(item["name"])
                content = fetch_content_from_google(item['id'], item['mimeType'])
                if content:
                    source = f"{item['name']}:{item['id']}"
                    documents.append(Document(page_content=content, metadata={"id": item['id'], "source": source}))

        page_token = results.get('nextPageToken', None)
        if page_token is None:
            break

    # Now iterate through the results and recursively call the function for any folders found
    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            list_all_files_recursive(item['id'], documents)


def load_google_documents():
    """Loads Google Docs, Sheets, and Slides from Google Drive and extracts their content."""

    customers_folder_id = get_folder_id('Customers', shared_drive_id)

    documents = []
    list_all_files_recursive(customers_folder_id, documents)
    return documents


def fetch_content_from_google(file_id, mime_type):
    """Fetches content from Google Drive files based on their type."""

    # Set a timeout for socket operations (e.g., 30 seconds)
    socket.setdefaulttimeout(30)  

    max_retries = 3
    retry_delay = 1  # Start with 1 second delay

    if mime_type == 'application/vnd.google-apps.document':
        for attempt in range(max_retries):
            try:
                request = drive_service.files().export(fileId=file_id, mimeType='text/plain')
                content = request.execute().decode('utf-8')
                return content
            except (HttpError, socket.timeout) as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for Google Doc {file_id}: {e}")
                if attempt < max_retries - 1:  # Don't wait after the last attempt
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the delay for the next attempt
                else:
                    print(f"An error occurred while fetching content from Google Doc {file_id}: {e}")
                    return None

    elif mime_type == 'application/vnd.google-apps.spreadsheet':
        for attempt in range(max_retries):
            try:
                sheet_metadata = sheets_service.spreadsheets().get(spreadsheetId=file_id).execute()
                sheets = sheet_metadata.get('sheets', '')
                sheet_name = sheets[0]['properties']['title'] if sheets else 'Sheet1'
                print(f"Using sheet name: {sheet_name}")

                result = sheets_service.spreadsheets().values().get(
                    spreadsheetId=file_id, range=f"{sheet_name}!A1:Z1000"
                ).execute()
                values = result.get('values', [])
                return "\n".join([", ".join(row) for row in values])

            except (HttpError, socket.timeout) as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for Google Sheet {file_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"Failed to retrieve metadata for Google Sheet {file_id}: {e}")
                    return None

    elif mime_type == 'application/vnd.google-apps.presentation':
        for attempt in range(max_retries):
            try:
                slides_service = build('slides', 'v1', credentials=creds)
                presentation = slides_service.presentations().get(presentationId=file_id).execute()
                slides = presentation.get('slides', [])
                content = ""
                for slide in slides:
                    for element in slide.get('pageElements', []):
                        if 'shape' in element and 'text' in element['shape']:
                            text_elements = element['shape']['text']['textElements']
                            for text_element in text_elements:
                                if 'textRun' in text_element:
                                    content += text_element['textRun']['content']
                return content
            except (HttpError, socket.timeout) as e:
                print(f"Attempt {attempt + 1}/{max_retries} failed for Google Slide {file_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"An error occurred while fetching content from Google Slide {file_id}: {e}")
                    return None
    else:
        print(f"Unsupported mime type: {mime_type}")
        return None

def split_documents(documents):
    """Splits large documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks):
    """Adds the split document chunks to Chroma DB and removes orphaned documents."""
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(), client=client
    )
    # db = client.get_or_create_collection(name="embed", embedding_function
    # get_embedding_function())

    # Retrieve existing document IDs from Chroma DB
    existing_items = db.get(include=[])  # Get existing document IDs
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Calculate the new chunks to be added
    chunks_with_ids = calculate_chunk_ids(chunks)
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    # chunks_content = [chunk.page_content for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    # chunks_metadata = [chunk.metadata for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]
    # chunks_content = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]


    # Add new documents to Chroma
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        # db.add(documents=chunks_content, metadatas=chunks_metadata, ids=new_chunk_ids)
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        print("✅ Documents added to Chroma DB.")
    else:
        print("✅ No new documents to add")

    # Check if documents exist in Chroma after addition
    existing_after_addition = db.get(include=[])
    print(f"Number of documents in DB after addition: {len(existing_after_addition['ids'])}")

    # Retrieve current Google Drive file IDs
    drive_file_ids = get_google_drive_file_ids()
    print(f"Google Drive IDs: {drive_file_ids}")

    # Find orphaned IDs
    print(f"File Names: {existing_ids}")
    existing_ids = {i.split(":")[1] for i in existing_ids}
    print(f"New Existing IDs: {existing_ids}")
    print(len(existing_ids),len(drive_file_ids))
    orphaned_ids = existing_ids - drive_file_ids  # Find IDs in DB but not in Google Drive
    # Filter out values that are not 'None'

    print(f"Identified orphaned IDs: {orphaned_ids}")

    if orphaned_ids:
      print(f"🗑 Found orphaned documents: {len(orphaned_ids)}")
      all_ids = get_all_ids(db)
      print("List of all IDs in the Chroma database:")
      print(all_ids)

      # Using list comprehension to find orphaned document IDs in the database
      orphaned_doc_ids_in_db = [i for i in all_ids for j in orphaned_ids if j in i]

      # Printing the orphaned document IDs
      for orphaned_id in orphaned_doc_ids_in_db:
          print(f"Orphaned Doc id: {orphaned_id}")
      delete_orphaned_documents(db,orphaned_doc_ids_in_db)

    else:
        print("✅ No orphaned documents to remove")


def get_all_ids(db):
    """
    Retrieves all document IDs from the Chroma database.

    Parameters:
        db (Chroma): The Chroma database instance.

    Returns:
        list: A list of document IDs in the Chroma DB.
    """
    try:
        # Retrieve all documents and their metadata from the Chroma DB
        all_documents = db.get(include=["documents", "metadatas"])

        # Extract the document IDs from the metadata
        all_ids = [doc['id'] for doc in all_documents['metadatas']]
        return all_ids

    except Exception as e:
        print(f"Error occurred while fetching IDs: {e}")
        return []

def find_document_by_id(db, doc_id):
    """
    Finds all entries in the Chroma DB associated with a given document ID.

    Parameters:
        db (Chroma): The Chroma database instance.
        doc_id (str): The document ID to search for.

    Returns:
        list: A list of entries associated with the provided ID, or an empty list if not found.
    """
    try:
        # Query the Chroma DB for the specific document ID
        result = db.get(include=["documents", "metadatas"])

        # Filter documents by the specific ID
        matching_documents = [
            doc for doc, meta in zip(result['documents'], result['metadatas'])
            if meta['id'] == doc_id
        ]

        return matching_documents

    except Exception as e:
        print(f"Error occurred while fetching document with ID {doc_id}: {e}")
        return []

def delete_orphaned_documents(db, orphaned_ids):
    """Deletes documents from the Chroma DB based on orphaned Google Drive file IDs."""
    if orphaned_ids:
        print(f"🗑 Deleting {len(orphaned_ids)} orphaned documents.")
        db.delete(ids=list(orphaned_ids))
        db.persist()
        print("✅ Orphaned documents deleted from Chroma DB.")
    else:
        print("✅ No orphaned documents to delete.")


def get_google_drive_file_ids():
    """Retrieves all Google Drive file IDs."""
    customers_folder_id = get_folder_id('Customers', shared_drive_id)
    documents = []
    list_all_files_recursive(customers_folder_id, documents)
    return {file.metadata['id'] for file in documents}  # Use a set for quick lookup


def clear_database():
    """Clears the Chroma database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print("Chroma path found")
    else:
      print("Chroma path not found")
    print("✨ Chroma database cleared.")

if __name__ == "__main__":
    main()

# --- Code to be executed after main() ---

def get_one_day_back_timestamp():
    """Returns a timestamp representing the date and time 24 hours ago."""
    one_day_back = datetime.now() - timedelta(days=1)
    return one_day_back.isoformat() + 'Z'  # Convert to RFC3339 format for comparison with modifiedTime

def count_updated_files_in_past_day():
    """Checks which files were updated in the past day and returns a list of tuples (file_id, mime_type)."""
    one_day_back_timestamp = get_one_day_back_timestamp()

    # Fetch all Google Drive files (Google Docs, Sheets, and Slides)
    results = drive_service.files().list(
        q="mimeType='application/vnd.google-apps.spreadsheet' or mimeType='application/vnd.google-apps.document' or mimeType='application/vnd.google-apps.presentation'",
        fields="files(id, name, mimeType, modifiedTime)"
    ).execute()

    files = results.get('files', [])
    updated_files = []

    if not files:
        print('No files found in Google Drive.')
    else:
        # Iterate over the files and check if they were updated in the last 24 hours
        for file in files:
            modified_time = file['modifiedTime']

            # Compare the modifiedTime with one_day_back_timestamp
            if modified_time > one_day_back_timestamp:
                updated_files.append((file['id'], file['mimeType']))  # Append ID and MIME type

    print(f"Total files updated in the past 24 hours: {len(updated_files)}")
    return updated_files

def update_db_with_updated_files(db):
    """Updates the Chroma DB with updated Google Drive files."""

    # Step 1: Get the list of files updated in the past 24 hours
    updated_files = count_updated_files_in_past_day()

    if not updated_files:
        print("✅ No updated files to process.")
        return

    print(f"🔄 Updating Chroma DB with {len(updated_files)} updated files.")

    # Step 2: For each updated file, remove existing entries from Chroma DB and re-add them
    for file_id, mime_type in updated_files:
        try:
            # Remove existing entries for the file
            print(f"🗑 Removing old entries for file ID: {file_id}")
            all_ids = get_all_ids(db)
            orphaned_doc_ids_in_db = [i for i in all_ids if file_id in i]
            delete_orphaned_documents(db, orphaned_doc_ids_in_db)

            # Fetch the updated content from Google Drive
            print(f"📄 Fetching updated content for file ID: {file_id}")
            file_content = fetch_content_from_google(file_id, mime_type)  # Use the file's MIME type to fetch content

            if file_content is None:
                print(f"⚠ No content fetched for file ID: {file_id}. Skipping...")
                continue

            # Convert the content into a Document object with metadata
            document = Document(page_content=file_content, metadata={"id": file_id, "source": f"{file_id}:{mime_type}"})

            # Split the document content into chunks (if needed)
            print(f"✂ Splitting content into chunks for file ID: {file_id}")
            chunks = split_documents([document])  # Make sure `split_documents` returns chunks as Document objects

            # Add the updated chunks to Chroma
            print(f"➕ Adding updated content for file ID: {file_id} into Chroma DB")
            add_updated_file_to_chroma(db,chunks)  # Pass the chunks to add_to_chroma function

            print(f"✅ Successfully updated file ID: {file_id} in Chroma DB.")

        except Exception as e:
            print(f"❌ Error processing file ID {file_id}: {e}")

    print("🔄 Database update process completed.")

def add_updated_file_to_chroma(db,chunks):
    # Retrieve existing document IDs from Chroma DB
    existing_items = db.get(include=[])  # Get existing document IDs
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Calculate the new chunks to be added
    chunks_with_ids = calculate_chunk_ids(chunks)
    new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["id"] not in existing_ids]

    # Add new documents to Chroma
    if new_chunks:
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        print("✅ Documents added to Chroma DB.")
    else:
        print("✅ No new documents to add")

    # Check if documents exist in Chroma after addition
    existing_after_addition = db.get(include=[])
    print(f"Number of documents in DB after addition: {len(existing_after_addition['ids'])}")

# Create the Chroma database instance
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function(), client=client)

# Update the database with updated files
update_db_with_updated_files(db)

# Fetch and print all documents and their metadata
all_documents = db.get(include=["documents", "metadatas"])
for doc, metadata in zip(all_documents["documents"], all_documents["metadatas"]):
    print(f"Document: {doc}")
    print(f"Metadata: {metadata}")
    print("-" * 40)