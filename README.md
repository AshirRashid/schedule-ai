# Introduction
This tool helps you identify which of your emails contain event information, so you don't have to manually search through them.

# Design
A simple RAG pipeline is used to index the emails. This prompt is then used to retrieve emails with event information:
`get all possible chunks that look like they are referring to a possible event`.

# Running
This code has been tested on Python 3.11

Install dependencies using `pip install -r requirements.txt`

Get you access [Google access credentials](https://developers.google.com/workspace/guides/create-credentials) and place then in the current directory and spefiy the name using the `globals.py` file with the `GCAL_CREDENTIALS_FILE` variable name.

Run `gmail_load_pipeline.py` to retrieve your emails, chunk them, and index them usign Chroma.

Run `query_db.py` to find out which of your emails contain event information.
