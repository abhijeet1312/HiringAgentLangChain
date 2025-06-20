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

from urllib.parse import urlparse
from fastapi import HTTPException
import warnings
warnings.filterwarnings("ignore")

from azurestorage import (
    azure_config
)

load_dotenv()

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

# ---------- Request Model ----------
class URLData(BaseModel):
    resumes: List[str]
    job_descriptions: List[str]
    voice_interview_threshold: Optional[float] = 3.0

# ---------- Main Logic Function ----------



# async def screen_candidates_from_urls_logic(payload: URLData):
#     """
#     Main logic for screening candidates from URLs
#     This function contains all the business logic from the original endpoint
#     """
#     if not s3_client:
#         raise HTTPException(status_code=500, detail="S3 client not initialized.")

#     resume_urls = payload.resumes
#     job_desc_urls = payload.job_descriptions
#     voice_interview_threshold = payload.voice_interview_threshold

#     if not resume_urls and not job_desc_urls:
#         raise HTTPException(status_code=400, detail="No resume or job description URLs provided.")
#     if not job_desc_urls:
#         raise HTTPException(status_code=400, detail="At least one job description URL is required.")

#     job_desc_s3_info = parse_s3_url_to_bucket_key(job_desc_urls[0])
#     resume_s3_info_list = []
#     for resume_url in resume_urls:
#         try:
#             s3_info = parse_s3_url_to_bucket_key(resume_url)
#             s3_info['original_url'] = resume_url
#             resume_s3_info_list.append(s3_info)
#         except Exception as e:
#             print(f"Failed to parse resume URL {resume_url}: {str(e)}")

#     if not resume_s3_info_list:
#         raise HTTPException(status_code=400, detail="No valid resume S3 URLs could be parsed")

#     job_desc_text = extract_text_from_s3(job_desc_s3_info['bucket'], job_desc_s3_info['key'])
#     if not job_desc_text.strip():
#         raise HTTPException(status_code=400, detail="Job description is empty")

#     processed_files = []
#     for i, resume_info in enumerate(resume_s3_info_list):
#         try:
#             resume_text = extract_text_from_s3(resume_info['bucket'], resume_info['key'])
#             if resume_text.strip():
#                 processed_files.append({
#                     'index': i,
#                     'bucket': resume_info['bucket'],
#                     'key': resume_info['key'],
#                     'text': resume_text,
#                     'filename': resume_info['key'].split('/')[-1],
#                     'original_url': resume_info.get('original_url', '')
#                 })
#         except Exception as e:
#             print(f"Error processing resume {resume_info['key']}: {e}")

#     if not processed_files:
#         raise HTTPException(status_code=400, detail="No resumes could be processed")

#     agent = CandidateScreeningAgent(job_description=job_desc_text)

#     temp_files = []
#     try:
#         for file_info in processed_files:
#             with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
#                 temp_file.write(file_info['text'])
#                 temp_files.append(temp_file.name)
#                 file_info['temp_path'] = temp_file.name

#         assessments = agent.batch_screen_candidates(temp_files)
#         report_path = TEMP_DIR / "candidate_assessments.csv"
#         qualified_data = agent.generate_report(
#             assessments,
#             output_path=str(report_path),
#             voice_interview_threshold=voice_interview_threshold
#         )
#         cleanup_results = await cleanup_s3_files_after_processing(resume_s3_info_list + [job_desc_s3_info])

#     finally:
#         for temp_file in temp_files:
#             try:
#                 os.unlink(temp_file)
#             except Exception as e:
#                 print(f"Error deleting temp file {temp_file}: {e}")

#     response_data = {
#         "timestamp": datetime.now().isoformat(),
#         "job_description_s3": job_desc_s3_info,
#         "resume_s3_info": resume_s3_info_list,
#         "job_description_preview": job_desc_text[:500] + "..." if len(job_desc_text) > 500 else job_desc_text,
#         "job_description_length": len(job_desc_text),
#         "total_candidates": len(assessments),
#         "successfully_processed": len(processed_files),
#         "failed_processing": len(resume_s3_info_list) - len(processed_files),
#         "assessments": assessments,
#         "qualified_candidates": qualified_data.get('qualified_candidates', []),
#         "total_qualified": qualified_data.get('total_qualified', 0),
#         "email_recipients": qualified_data.get('email_recipients', []),
#         "voice_interview_threshold": voice_interview_threshold,
#         "report_generated": True,
#         "s3_cleanup": cleanup_results
#     }

#     if qualified_data.get('qualified_candidates'):
#         try:
#             voice_results = agent.trigger_voice_interviews_for_qualified(qualified_data)
#             response_data["voice_interviews"] = {
#                 "initiated": True,
#                 "results": voice_results
#             }
#         except Exception as voice_error:
#             response_data["voice_interviews"] = {
#                 "initiated": False,
#                 "error": str(voice_error)
#             }
#     else:
#         response_data["voice_interviews"] = {
#             "initiated": False,
#             "reason": "No candidates qualified"
#         }

#     return JSONResponse(status_code=200, content={
#         "success": True,
#         "message": "Screening completed",
#         "data": response_data
#     })


from fastapi.responses import JSONResponse
from fastapi import HTTPException
from datetime import datetime
import tempfile, os
from pathlib import Path

TEMP_DIR = Path(tempfile.gettempdir())

async def screen_candidates_from_urls_logic(payload: URLData):
    """
    Main logic for screening candidates from Azure Blob URLs
    """
    resume_urls = payload.resumes
    job_desc_urls = payload.job_descriptions
    voice_interview_threshold = payload.voice_interview_threshold

    if not resume_urls and not job_desc_urls:
        raise HTTPException(status_code=400, detail="No resume or job description URLs provided.")
    if not job_desc_urls:
        raise HTTPException(status_code=400, detail="At least one job description URL is required.")

    # --- Parse Azure Blob URLs ---
    try:
        job_desc_blob_info = parse_azure_url_to_container_blob_path(job_desc_urls[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job description URL: {str(e)}")

    resume_blob_info_list = []
    for resume_url in resume_urls:
        try:
            blob_info = parse_azure_url_to_container_blob_path(resume_url)
            blob_info['original_url'] = resume_url
            resume_blob_info_list.append(blob_info)
        except Exception as e:
            print(f"Failed to parse resume URL {resume_url}: {str(e)}")

    if not resume_blob_info_list:
        raise HTTPException(status_code=400, detail="No valid resume Azure Blob URLs could be parsed")

    # --- Extract job description text ---
    job_desc_text = extract_text_from_azure(job_desc_blob_info['blob_path'])
    if not job_desc_text.strip():
        raise HTTPException(status_code=400, detail="Job description is empty")

    # --- Process resumes ---
    processed_files = []
    for i, resume_info in enumerate(resume_blob_info_list):
        try:
            resume_text = extract_text_from_azure(resume_info['blob_path'])
            if resume_text.strip():
                processed_files.append({
                    'index': i,
                    'container': resume_info['container'],
                    'blob_path': resume_info['blob_path'],
                    'text': resume_text,
                    'filename': resume_info['blob_path'].split('/')[-1],
                    'original_url': resume_info.get('original_url', '')
                })
        except Exception as e:
            print(f"Error processing resume {resume_info['blob_path']}: {e}")

    if not processed_files:
        raise HTTPException(status_code=400, detail="No resumes could be processed")

    agent = CandidateScreeningAgent(job_description=job_desc_text)

    # --- Save texts to temp files ---
    temp_files = []
    try:
        for file_info in processed_files:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
                temp_file.write(file_info['text'])
                temp_files.append(temp_file.name)
                file_info['temp_path'] = temp_file.name

        assessments = agent.batch_screen_candidates(temp_files)
        report_path = TEMP_DIR / "candidate_assessments.csv"
        qualified_data = agent.generate_report(
            assessments,
            output_path=str(report_path),
            voice_interview_threshold=voice_interview_threshold
        )

        # --- Cleanup Azure blobs after processing ---
        cleanup_results = await cleanup_azure_files_after_processing(resume_blob_info_list + [job_desc_blob_info])

    finally:
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                print(f"Error deleting temp file {temp_file}: {e}")

    # --- Final Response ---
    response_data = {
        "timestamp": datetime.now().isoformat(),
        "job_description_blob": job_desc_blob_info,
        "resume_blob_info": resume_blob_info_list,
        "job_description_preview": job_desc_text[:500] + "..." if len(job_desc_text) > 500 else job_desc_text,
        "job_description_length": len(job_desc_text),
        "total_candidates": len(assessments),
        "successfully_processed": len(processed_files),
        "failed_processing": len(resume_blob_info_list) - len(processed_files),
        "assessments": assessments,
        "qualified_candidates": qualified_data.get('qualified_candidates', []),
        "total_qualified": qualified_data.get('total_qualified', 0),
        "email_recipients": qualified_data.get('email_recipients', []),
        "voice_interview_threshold": voice_interview_threshold,
        "report_generated": True,
        "azure_cleanup": cleanup_results
    }

    if qualified_data.get('qualified_candidates'):
        try:
            voice_results = agent.trigger_voice_interviews_for_qualified(qualified_data)
            response_data["voice_interviews"] = {
                "initiated": True,
                "results": voice_results
            }
        except Exception as voice_error:
            response_data["voice_interviews"] = {
                "initiated": False,
                "error": str(voice_error)
            }
    else:
        response_data["voice_interviews"] = {
            "initiated": False,
            "reason": "No candidates qualified"
        }

    return JSONResponse(status_code=200, content={
        "success": True,
        "message": "Screening completed",
        "data": response_data
    })


# ---------- Helper Functions ----------
# def extract_text_from_s3(bucket: str, key: str) -> str:
#     """Extract text from PDF/Word/Text stored in S3."""
#     try:
#         print(f"Getting object from S3: {bucket}/{key}")
#         response = s3_client.get_object(Bucket=bucket, Key=key)
#         file_content = response['Body'].read()
#         content_type = response.get('ContentType', '').lower()

#         # PDF
#         if 'pdf' in content_type or key.lower().endswith('.pdf'):
#             try:
#                 import pdfplumber
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#                     temp_file.write(file_content)
#                     temp_file_path = temp_file.name

#                 with pdfplumber.open(temp_file_path) as pdf:
#                     text = ""
#                     for page in pdf.pages:
#                         text += page.extract_text() or ""

#                 os.remove(temp_file_path)
#                 return text

#             except ImportError:
#                 raise HTTPException(
#                     status_code=500,
#                     detail="pdfplumber not installed. Install it using: pip install pdfplumber"
#                 )

#         # Text
#         elif 'text' in content_type or key.lower().endswith('.txt'):
#             return file_content.decode('utf-8')

#         # Word (.docx)
#         elif 'word' in content_type or key.lower().endswith(('.doc', '.docx')):
#             try:
#                 import docx2txt
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
#                     temp_file.write(file_content)
#                     temp_file_path = temp_file.name

#                 text = docx2txt.process(temp_file_path)
#                 os.remove(temp_file_path)
#                 return text

#             except ImportError:
#                 raise HTTPException(
#                     status_code=500,
#                     detail="docx2txt not installed. Install it using: pip install docx2txt"
#                 )

#         # Fallback text decode
#         else:
#             try:
#                 return file_content.decode('utf-8')
#             except UnicodeDecodeError:
#                 raise HTTPException(
#                     status_code=400,
#                     detail=f"Unsupported file type for S3 object: {bucket}/{key}"
#                 )

#     except ClientError as e:
#         error_code = e.response['Error']['Code']
#         if error_code == 'NoSuchKey':
#             raise HTTPException(status_code=404, detail=f"S3 object not found: {bucket}/{key}")
#         elif error_code == 'NoSuchBucket':
#             raise HTTPException(status_code=404, detail=f"S3 bucket not found: {bucket}")
#         else:
#             raise HTTPException(status_code=500, detail=f"S3 error: {str(e)}")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error extracting text from S3 {bucket}/{key}: {e}")





def extract_text_from_azure(blob_path: str) -> str:
    """
    Extract text from PDF/Word/Text stored in Azure Blob Storage.
    :param blob_path: The blob path inside the container (e.g., 'resumes/file.pdf')
    """
    try:
        print(f"Fetching blob from Azure: {azure_config.container_name}/{blob_path}")
        blob_client = azure_config.container_client.get_blob_client(blob_path)
        blob_data = blob_client.download_blob().readall()

        # Guess content type from file extension
        if blob_path.lower().endswith('.pdf'):
            try:
                import pdfplumber
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(blob_data)
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

        elif blob_path.lower().endswith('.txt'):
            return blob_data.decode('utf-8')

        elif blob_path.lower().endswith(('.doc', '.docx')):
            try:
                import docx2txt
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp_file:
                    temp_file.write(blob_data)
                    temp_file_path = temp_file.name

                text = docx2txt.process(temp_file_path)
                os.remove(temp_file_path)
                return text

            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="docx2txt not installed. Install it using: pip install docx2txt"
                )

        # Fallback decode attempt
        else:
            try:
                return blob_data.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported or non-text file type for Azure blob: {blob_path}"
                )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from Azure blob {blob_path}: {str(e)}"
        )

    
    
    

# async def cleanup_s3_files_after_processing(s3_objects: list) -> dict:
#     """Delete files from S3 after processing."""
#     cleanup_results = {
#         "total_files": len(s3_objects),
#         "successfully_deleted": 0,
#         "failed_deletions": 0,
#         "errors": []
#     }
    
#     for s3_obj in s3_objects:
#         try:
#             if not isinstance(s3_obj, dict) or 'bucket' not in s3_obj or 'key' not in s3_obj:
#                 cleanup_results["errors"].append(f"Invalid S3 object info: {s3_obj}")
#                 cleanup_results["failed_deletions"] += 1
#                 continue
            
#             bucket = s3_obj['bucket']
#             key = s3_obj['key']
            
#             s3_client.delete_object(Bucket=bucket, Key=key)
            
#             print(f"Deleted {key} from bucket {bucket}")
#             cleanup_results["successfully_deleted"] += 1
            
#         except ClientError as e:
#             error_msg = f"Error deleting S3 object {s3_obj}: {str(e)}"
#             print(error_msg)
#             cleanup_results["errors"].append(error_msg)
#             cleanup_results["failed_deletions"] += 1
            
#         except Exception as e:
#             error_msg = f"Unexpected error deleting S3 object {s3_obj}: {str(e)}"
#             print(error_msg)
#             cleanup_results["errors"].append(error_msg)
#             cleanup_results["failed_deletions"] += 1
    
#     return cleanup_results


async def cleanup_azure_files_after_processing(azure_objects: list) -> dict:
    """Delete files from Azure Blob Storage after processing."""

    cleanup_results = {
        "total_files": len(azure_objects),
        "successfully_deleted": 0,
        "failed_deletions": 0,
        "errors": []
    }

    for azure_obj in azure_objects:
        try:
            if not isinstance(azure_obj, dict) or 'container' not in azure_obj or 'blob_path' not in azure_obj:
                cleanup_results["errors"].append(f"Invalid Azure object info: {azure_obj}")
                cleanup_results["failed_deletions"] += 1
                continue

            container = azure_obj['container']
            blob_path = azure_obj['blob_path']

            blob_service_client = azure_config.blob_service_client.from_connection_string(azure_config.connection_string)
            container_client = blob_service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(blob_path)

            blob_client.delete_blob()

            print(f"Deleted {blob_path} from container {container}")
            cleanup_results["successfully_deleted"] += 1

        except Exception as e:
            error_msg = f"Error deleting Azure blob {azure_obj}: {str(e)}"
            print(error_msg)
            cleanup_results["errors"].append(error_msg)
            cleanup_results["failed_deletions"] += 1

    return cleanup_results




# def parse_s3_url_to_bucket_key(s3_url: str) -> dict:
#     """Parse S3 URL to extract bucket and key information."""
#     try:
#         from urllib.parse import urlparse
        
#         parsed = urlparse(s3_url)
        
#         # Handle S3 URL format: https://bucket-name.s3.region.amazonaws.com/path/to/file
#         if 's3.' in parsed.netloc and 'amazonaws.com' in parsed.netloc:
#             bucket = parsed.netloc.split('.')[0]
#             key = parsed.path.lstrip('/')
#         else:
#             raise ValueError(f"Unrecognized S3 URL format: {s3_url}")
        
#         return {"bucket": bucket, "key": key}
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Error parsing S3 URL {s3_url}: {str(e)}"
#         )




def parse_azure_url_to_container_blob_path(azure_url: str) -> dict:
    """
    Parse Azure Blob Storage URL to extract container and blob path.

    Example URL:
    https://<account>.blob.core.windows.net/<container>/<folder>/<filename>
    """
    try:
        parsed = urlparse(azure_url)

        # Example netloc: aurjobsaiagentsstorage.blob.core.windows.net
        # Example path: /aurjobs123/job_descriptions/file.pdf

        path_parts = parsed.path.lstrip('/').split('/', 1)

        if len(path_parts) != 2:
            raise ValueError("URL does not contain a container and blob path")

        container_name = path_parts[0]
        blob_path = path_parts[1]

        return {
            "container": container_name,
            "blob_path": blob_path
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error parsing Azure Blob URL {azure_url}: {str(e)}"
        )

