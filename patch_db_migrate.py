from pymongo import MongoClient

def migrate_db():
    print("Connecting to Local MongoDB...")
    local_client = MongoClient("mongodb://127.0.0.1:27017/")
    
    print("Connecting to MongoDB Atlas...")
    atlas_client = MongoClient("mongodb+srv://bayshoregraphicsbd_db_user:OKkShkcnhiJ6Fyiw@cluster0.s4cjul1.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    
    # We will grab all data from the database 'ai_chatbot_db'
    db_name = "ai_chatbot_db"
    
    local_db = local_client[db_name]
    atlas_db = atlas_client[db_name]
    
    collections = local_db.list_collection_names()
    print(f"Found {len(collections)} collections to migrate: {collections}")
    
    for coll_name in collections:
        local_coll = local_db[coll_name]
        atlas_coll = atlas_db[coll_name]
        
        # Read all documents from local
        docs = list(local_coll.find({}))
        
        if len(docs) > 0:
            print(f"Migrating collection '{coll_name}'. Found {len(docs)} documents.")
            
            # Wipe existing docs on atlas to avoid duplicates
            atlas_coll.delete_many({})
            
            # Insert into atlas
            atlas_coll.insert_many(docs)
            print(f"[OK] Successfully inserted {len(docs)} documents into '{coll_name}' on Atlas.")
        else:
            print(f"[!] Collection '{coll_name}' is empty. Skipping.")

    print("\n>>> Database migration fully completed!")

if __name__ == "__main__":
    migrate_db()
