import boto3
import sqlite3
import os
from datetime import datetime

# CONFIGURATION
BUCKET_NAME = "thesis-data-ab3rnhard"
DB_PATH = "/home/AB3rnhard/DABE_Thesis_Hoefer_Khanh/Data/polymarket_gamma_dynamic.sqlite"
BACKUP_PATH = "/home/AB3rnhard/DABE_Thesis_Hoefer_Khanh/Data/s3_upload_tmp.sqlite"
S3_KEY = f"backups/db_{datetime.now().strftime('%Y-%m-%d')}.sqlite"

def sync():
    # 1. Create a safe backup of the live DB
    print("Creating safe backup...")
    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(BACKUP_PATH)
    src.backup(dst)
    dst.close()
    src.close()

    # 2. Upload to S3
    print(f"Uploading to S3: {S3_KEY}...")
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="eu-north-1"
    )
    
    s3.upload_file(BACKUP_PATH, BUCKET_NAME, S3_KEY)
    print("Upload complete!")

    # 3. Clean up the temp file
    os.remove(BACKUP_PATH)

if __name__ == "__main__":
    sync()