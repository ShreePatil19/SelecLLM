import os
import json
import uuid

CHAT_DIR = "chats"
os.makedirs(CHAT_DIR, exist_ok=True)

def _chat_file(username, chat_id):
    return os.path.join(CHAT_DIR, f"{username}__{chat_id}.json")

def create_new_chat(username):
    chat_id = str(uuid.uuid4())
    with open(_chat_file(username, chat_id), "w") as f:
        json.dump([], f)
    return chat_id

def save_message(username, chat_id, role, message):
    history = load_history(username, chat_id)
    history.append({"role": role, "message": message})
    with open(_chat_file(username, chat_id), "w") as f:
        json.dump(history, f)

def load_history(username, chat_id):
    path = _chat_file(username, chat_id)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)

def get_chat_titles(username):
    files = os.listdir(CHAT_DIR)
    user_files = [f for f in files if f.startswith(f"{username}__")]
    return [f.split("__")[1].replace(".json", "") for f in user_files]

def get_chat_by_id(username, chat_id):
    return load_history(username, chat_id)
