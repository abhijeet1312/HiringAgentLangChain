# endpoint2.py
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import tempfile
import os
import json as json_lib
from datetime import datetime
from pathlib import Path
from screening import CandidateScreeningAgent
from dotenv import load_dotenv





# ---------- Constants ----------
TEMP_DIR = Path("temp_resumes")
TEMP_DIR.mkdir(exist_ok=True)

# ---------- AWS S3 Client ----------
try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY'),
        aws_secret_access_key=os.getenv('AWS_SECRET_KEY'),
        region_name=os.getenv('AWS_REGION', 'ap-south-1')
    )
except NoCredentialsError:
    print("AWS credentials not found.")
    s3_client = None

# ---------- Helper Functions ----------
def extract_text_from_s3(bucket: str, key: str) -> str:
    """Extract text from PDF/Word/Text stored in S3."""
    try:
        print(f"Getting object from S3: {bucket}/{key}")
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_content = response['Body'].read()
        content_type = response.get('ContentType', '').lower()

        # PDF
        if 'pdf' in content_type or key.lower().endswith('.pdf'):
            try:
                import pdfplumber
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name

                with pdfplumber.open(temp_file_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""

                os.remove(temp_file_path)
                return text

            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="pdfplumber not installed. Install it using: pip install pdfplumber"
                )

        # Text
        elif 'text' in content_type or key.lower().endswith('.txt'):
            return file_content.decode('utf-8')

        # Word (.docx)
        elif 'word' in content_type or key.lower().endswith(('.doc', '.docx')):
            try:
                import docx2txt
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                    temp_file.write(file_content)
                    temp_file_path = temp_file.name

                text = docx2txt.process(temp_file_path)
                os.remove(temp_file_path)
                return text

            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="docx2txt not installed. Install it using: pip install docx2txt"
                )

        # Fallback text decode
        else:
            try:
                return file_content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type for S3 object: {bucket}/{key}"
                )

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'NoSuchKey':
            raise HTTPException(status_code=404, detail=f"S3 object not found: {bucket}/{key}")
        elif error_code == 'NoSuchBucket':
            raise HTTPException(status_code=404, detail=f"S3 bucket not found: {bucket}")
        else:
            raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from S3 {bucket}/{key}: {e}")

async def cleanup_s3_files_after_processing(s3_objects: list) -> dict:
    """Delete files from S3 after processing."""
    cleanup_results = {
        "total_files": len(s3_objects),
        "successfully_deleted": 0,
        "failed_deletions": 0,
        "errors": []
    }
    
    for s3_obj in s3_objects:
        try:
            if not isinstance(s3_obj, dict) or 'bucket' not in s3_obj or 'key' not in s3_obj:
                cleanup_results["errors"].append(f"Invalid S3 object info: {s3_obj}")
                cleanup_results["failed_deletions"] += 1
                continue
            
            bucket = s3_obj['bucket']
            key = s3_obj['key']
            
            s3_client.delete_object(Bucket=bucket, Key=key)
            
            print(f"Deleted {key} from bucket {bucket}")
            cleanup_results["successfully_deleted"] += 1
            
        except ClientError as e:
            error_msg = f"Error deleting S3 object {s3_obj}: {str(e)}"
            print(error_msg)
            cleanup_results["errors"].append(error_msg)
            cleanup_results["failed_deletions"] += 1
            
        except Exception as e:
            error_msg = f"Unexpected error deleting S3 object {s3_obj}: {str(e)}"
            print(error_msg)
            cleanup_results["errors"].append(error_msg)
            cleanup_results["failed_deletions"] += 1
    
    return cleanup_results

def parse_s3_url_to_bucket_key(s3_url: str) -> dict:
    """Parse S3 URL to extract bucket and key information."""
    try:
        from urllib.parse import urlparse
        
        parsed = urlparse(s3_url)
        
        # Handle S3 URL format: https://bucket-name.s3.region.amazonaws.com/path/to/file
        if 's3.' in parsed.netloc and 'amazonaws.com' in parsed.netloc:
            bucket = parsed.netloc.split('.')[0]
            key = parsed.path.lstrip('/')
        else:
            raise ValueError(f"Unrecognized S3 URL format: {s3_url}")
        
        return {"bucket": bucket, "key": key}
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error parsing S3 URL {s3_url}: {str(e)}"
        )
