# endpoint2.py
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
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


# ---------- Request Model ----------
class URLData(BaseModel):
    resumes: List[str]
    job_descriptions: List[str]
    voice_interview_threshold: Optional[float] = 3.0

# ---------- Main Logic Function ----------




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




import tempfile
import os
import re
from urllib.parse import unquote
from fastapi import HTTPException

def extract_text_from_azure(blob_path: str) -> str:
    """
    Extract text from PDF/Word/Text stored in Azure Blob Storage with improved handling.
    :param blob_path: The blob path inside the container (e.g., 'resumes/file.pdf')
    """
    try:
        # Decode URL-encoded characters in the blob path
        decoded_blob_path = unquote(blob_path)
        print(f"Original blob path: {blob_path}")
        print(f"Decoded blob path: {decoded_blob_path}")
        print(f"Fetching blob from Azure: {azure_config.container_name}/{decoded_blob_path}")
        
        blob_client = azure_config.container_client.get_blob_client(decoded_blob_path)
        blob_data = blob_client.download_blob().readall()
        
        # Extract text based on file type
        if decoded_blob_path.lower().endswith('.pdf'):
            text = extract_pdf_text(blob_data)
        elif decoded_blob_path.lower().endswith('.txt'):
            text = extract_txt_text(blob_data)
        elif decoded_blob_path.lower().endswith(('.doc', '.docx')):
            text = extract_docx_text(blob_data)
        else:
            # Fallback decode attempt
            try:
                text = blob_data.decode('utf-8')
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        text = blob_data.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unable to decode file: {decoded_blob_path}"
                    )
        
        # Clean and normalize the extracted text
        cleaned_text = clean_extracted_text(text)
        return cleaned_text
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract text from Azure blob {decoded_blob_path if 'decoded_blob_path' in locals() else blob_path}: {str(e)}"
        )

def extract_pdf_text(blob_data: bytes) -> str:
    """Extract text from PDF using multiple methods in order of preference."""
    extraction_methods = [
        ("pdfminer.six (simple)", extract_pdf_with_pdfminer),
        ("pdfminer.six (advanced)", extract_pdf_with_pdfminer_advanced),
        ("PyPDF2 (fallback)", extract_pdf_with_pypdf2)
    ]
    
    for method_name, method_func in extraction_methods:
        try:
            print(f"Trying {method_name}...")
            text = method_func(blob_data)
            
            # Check if extraction was successful
            if text and text.strip() and len(text.strip()) > 10:
                print(f"Successfully extracted text using {method_name}")
                return text
            else:
                print(f"{method_name} returned empty or insufficient text")
                
        except ImportError as e:
            print(f"{method_name} not available: {str(e)}")
            continue
        except Exception as e:
            print(f"{method_name} failed: {str(e)}")
            continue
    
    # If all methods fail
    raise HTTPException(
        status_code=500,
        detail="All PDF extraction methods failed. Please check if the PDF is valid and readable."
    )

def extract_pdf_with_pdfminer(blob_data: bytes) -> str:
    """Extract text using pdfminer.six - primary method."""
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams
        import io
        
        # Create layout parameters for better text extraction
        laparams = LAParams(
            boxes_flow=0.5,      # Controls how text boxes are grouped
            word_margin=0.1,     # Word separation threshold
            char_margin=2.0,     # Character separation threshold
            line_margin=0.5,     # Line separation threshold
            detect_vertical=False # Don't detect vertical text
        )
        
        # Extract text with custom parameters
        text = extract_text(
            io.BytesIO(blob_data),
            laparams=laparams,
            maxpages=0,          # Process all pages
            password="",         # No password
            caching=True         # Enable caching for performance
        )
        
        return text
        
    except ImportError:
        raise ImportError("pdfminer.six not installed")
    except Exception as e:
        print(f"pdfminer extraction failed: {str(e)}")
        return ""

def extract_pdf_with_pdfminer_advanced(blob_data: bytes) -> str:
    """Advanced pdfminer extraction with more control - alternative method."""
    try:
        from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
        from pdfminer.pdfpage import PDFPage
        from pdfminer.converter import TextConverter
        from pdfminer.layout import LAParams
        import io
        
        # Set up PDF processing
        rsrcmgr = PDFResourceManager()
        output_string = io.StringIO()
        
        # Configure layout parameters for resume parsing
        laparams = LAParams(
            boxes_flow=0.5,
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            detect_vertical=False,
            all_texts=False
        )
        
        device = TextConverter(rsrcmgr, output_string, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        
        # Process each page
        for page in PDFPage.get_pages(
            io.BytesIO(blob_data),
            caching=True
        ):
            interpreter.process_page(page)
        
        text = output_string.getvalue()
        device.close()
        output_string.close()
        
        return text
        
    except ImportError:
        raise ImportError("pdfminer.six not installed")
    except Exception as e:
        print(f"Advanced pdfminer extraction failed: {str(e)}")
        return ""

def extract_pdf_with_pypdf2(blob_data: bytes) -> str:
    """Fallback PDF extraction using PyPDF2."""
    try:
        import PyPDF2
        import io
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(blob_data))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except ImportError:
        return ""

def extract_txt_text(blob_data: bytes) -> str:
    """Extract text from TXT files with encoding detection."""
    # Try multiple encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return blob_data.decode(encoding)
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, use errors='ignore'
    return blob_data.decode('utf-8', errors='ignore')

def extract_docx_text(blob_data: bytes) -> str:
    """Extract text from DOCX files."""
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

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text for better parsing."""
    if not text:
        return ""
    
    # Replace problematic Unicode characters with ASCII equivalents
    replacements = {
        '•': '- ',           # Bullet point
        '◦': '- ',           # White bullet
        '§': '',             # Section sign
        'ï': '',             # Various accented characters
        '–': '-',            # En dash
        '—': '-',            # Em dash
        ''': "'",            # Smart quotes
        ''': "'",
        '"': '"',
        '"': '"',
        '…': '...',          # Ellipsis
        '\u00a0': ' ',       # Non-breaking space
        '\u2022': '- ',      # Bullet point
        '\u25e6': '- ',      # White bullet
        '\u2013': '-',       # En dash
        '\u2014': '-',       # Em dash
        '\ufb01': 'fi',      # fi ligature - THIS IS THE KEY FIX
        '\ufb02': 'fl',      # fl ligature
        '\ufb03': 'ffi',     # ffi ligature
        '\ufb04': 'ffl',     # ffl ligature
        '\u2018': "'",       # Left single quote
        '\u2019': "'",       # Right single quote
        '\u201c': '"',       # Left double quote
        '\u201d': '"',       # Right double quote
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = text.strip()
    
    # Final cleanup - remove any remaining problematic characters
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    
    # Ensure text is not empty after cleaning
    if not text or len(text.strip()) < 10:
        raise ValueError("Extracted text is empty or too short after cleaning")
    
    return text

# Additional helper function for debugging
def debug_text_extraction(blob_path: str) -> dict:
    """Debug function to analyze text extraction issues."""
    try:
        blob_client = azure_config.container_client.get_blob_client(blob_path)
        blob_data = blob_client.download_blob().readall()
        
        # Try to detect encoding
        import chardet
        detected = chardet.detect(blob_data[:1000])  # Check first 1000 bytes
        
        return {
            "file_size": len(blob_data),
            "detected_encoding": detected,
            "first_100_chars": str(blob_data[:100]),
            "file_extension": blob_path.split('.')[-1].lower()
        }
    except Exception as e:
        return {"error": str(e)}
    
    


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
            
            # Decode URL-encoded characters in the blob path (KEY FIX)
            decoded_blob_path = unquote(blob_path)
            print(f"Original blob path: {blob_path}")
            print(f"Decoded blob path: {decoded_blob_path}")
            blob_service_client = azure_config.blob_service_client.from_connection_string(azure_config.connection_string)
            container_client = blob_service_client.get_container_client(container)
            blob_client = container_client.get_blob_client(decoded_blob_path)

            blob_client.delete_blob()

            print(f"Deleted {decoded_blob_path} from container {container}")
            cleanup_results["successfully_deleted"] += 1

        except Exception as e:
            error_msg = f"Error deleting Azure blob {azure_obj}: {str(e)}"
            print(error_msg)
            cleanup_results["errors"].append(error_msg)
            cleanup_results["failed_deletions"] += 1

    return cleanup_results








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

