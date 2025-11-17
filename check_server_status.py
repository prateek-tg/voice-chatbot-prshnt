# #!/usr/bin/env python3
# """
# Quick script to check the status of the deployed server
# and diagnose common issues.
# """

# import sys
# import os

# def check_chromadb():
#     """Check if ChromaDB is initialized."""
#     print("🔍 Checking ChromaDB status...")
    
#     chroma_dir = "./chroma_db"
#     if not os.path.exists(chroma_dir):
#         print(f"❌ ChromaDB directory not found: {chroma_dir}")
#         print("   Run: python3 initialize_data.py --reset")
#         return False
    
#     print(f"✅ ChromaDB directory exists: {chroma_dir}")
    
#     # Check for database files
#     sqlite_file = os.path.join(chroma_dir, "chroma.sqlite3")
#     if os.path.exists(sqlite_file):
#         print(f"✅ Database file found: {sqlite_file}")
#     else:
#         print(f"⚠️ Database file not found: {sqlite_file}")
#         print("   Database might not be initialized properly")
#         return False
    
#     try:
#         from vectorstore.chromadb_client import ChromaDBClient
#         client = ChromaDBClient()
#         info = client.get_collection_info()
#         count = info.get('count', 0)
        
#         if count > 0:
#             print(f"✅ ChromaDB collection has {count} documents")
#             return True
#         else:
#             print(f"⚠️ ChromaDB collection is empty (0 documents)")
#             print("   Run: python3 initialize_data.py --reset")
#             return False
            
#     except Exception as e:
#         print(f"❌ Error accessing ChromaDB: {e}")
#         print("   This is likely the issue your server is experiencing")
#         return False

# def check_redis():
#     """Check if Redis is accessible."""
#     print("\n🔍 Checking Redis status...")
    
#     try:
#         from utils.redis_manager import RedisManager
#         redis_mgr = RedisManager()
        
#         # Try to ping Redis
#         redis_mgr.redis_client.ping()
#         print("✅ Redis is accessible and responding")
#         return True
        
#     except Exception as e:
#         print(f"❌ Redis connection failed: {e}")
#         print("   Make sure Redis is running: redis-server")
#         return False

# def check_data_file():
#     """Check if data file exists."""
#     print("\n🔍 Checking data file...")
    
#     data_file = "data/info.txt"
#     if os.path.exists(data_file):
#         size = os.path.getsize(data_file)
#         print(f"✅ Data file exists: {data_file} ({size} bytes)")
#         return True
#     else:
#         print(f"❌ Data file not found: {data_file}")
#         print("   Create this file with your knowledge base content")
#         return False

# def check_env_vars():
#     """Check if required environment variables are set."""
#     print("\n🔍 Checking environment variables...")
    
#     from dotenv import load_dotenv
#     load_dotenv()
    
#     required_vars = {
#         "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
#         "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
#         "REDIS_PORT": os.getenv("REDIS_PORT", "6379"),
#     }
    
#     all_good = True
#     for var_name, var_value in required_vars.items():
#         if var_value and not var_value.startswith("your-"):
#             print(f"✅ {var_name} is set")
#         else:
#             print(f"❌ {var_name} is not set or invalid")
#             all_good = False
    
#     return all_good

# def main():
#     """Run all status checks."""
#     print("=" * 60)
#     print("Server Status Check")
#     print("=" * 60)
    
#     checks = {
#         "Data File": check_data_file(),
#         "Environment Variables": check_env_vars(),
#         "Redis": check_redis(),
#         "ChromaDB": check_chromadb(),
#     }
    
#     print("\n" + "=" * 60)
#     print("Summary")
#     print("=" * 60)
    
#     for check_name, result in checks.items():
#         status = "✅ PASS" if result else "❌ FAIL"
#         print(f"{check_name}: {status}")
    
#     all_passed = all(checks.values())
    
#     print("\n" + "=" * 60)
#     if all_passed:
#         print("✅ All checks passed! Server should be working.")
#         print("\nTo start the server:")
#         print("  python3 socketio_server.py")
#     else:
#         print("❌ Some checks failed. Please fix the issues above.")
#         print("\nMost common fix:")
#         print("  python3 initialize_data.py --reset")
#     print("=" * 60)
    
#     return 0 if all_passed else 1

# if __name__ == "__main__":
#     sys.exit(main())
