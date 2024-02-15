import json
import requests

url="http://localhost:8000/stream_chat"
message = "What is UWB RANGING?"
data = {"content" : message,
        "queries" : [],
        "answers": []}

headers = {"Content-type" : "application/json"}
full_response = ''
with requests.post(url, data = json.dumps(data), headers = headers, stream= True) as r:
    for chunk in r.iter_content():
        decoded_chunk = chunk.decode('utf-8')
        print(decoded_chunk)  
        full_response += decoded_chunk 

print (full_response)