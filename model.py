#model.py
import json
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from huggingface_hub import InferenceClient
import os
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import List, Optional
print("jai mata di")


from azurestorage import (
    delete_file_from_azure,
    upload_file_to_azure,
    validate_pdf_file,
    generate_unique_filename,
    azure_config,
    ResumeUploadResponse,
    JobDescriptionUploadResponse
)
# Import the screening functions from endpoint2
from endpoint2 import screen_candidates_from_urls_logic, URLData

# Request model (remove prompt field)
class MatchRequest(BaseModel):
    resume: str
    job_desc: str
    prompt: str

app = FastAPI()
load_dotenv()  # Load from .env

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow all origins
    allow_credentials=False,  # ❌ Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/jmd")
async def jmd():
    return "Jai mata Di"
@app.post("/screen-candidates-from-urls/")
async def screen_candidates_from_urls(payload: URLData):
    """
    Screen candidates from S3 URLs - Route imported from endpoint2.py
    """
    return await screen_candidates_from_urls_logic(payload)

@app.post("/upload-resumes", response_model=ResumeUploadResponse)
async def upload_resumes(
    resumes: List[UploadFile] = File(..., description="Resume PDF files")
):
    """
    Upload resume PDF files to S3 bucket
    
    Args:
        resumes: List of resume PDF files
    
    Returns:
        JSON response with downloadable URLs for uploaded resume files
    """
    
    # Validate that files are provided
    if not resumes:
        raise HTTPException(
            status_code=400, 
            detail="Resume files are required"
        )
    
    # Validate file types
    for file in resumes:
        if not validate_pdf_file(file):
            raise HTTPException(
                status_code=400,
                detail=f"Only PDF files are allowed. Invalid file: {file.filename}"
            )
    
    # Check file size limits (5MB per file)
    max_file_size = 5 * 1024 * 1024  # 5MB
    for file in resumes:
        if file.size and file.size > max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds 5MB limit: {file.filename}"
            )
    
    # Check file count limits
    if len(resumes) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 resume files allowed"
        )
    
    uploaded_resume_urls = []
    
    try:
        # Upload resume files
        for resume_file in resumes:
            # Reset file pointer to beginning
            await resume_file.seek(0)
            url = await upload_file_to_azure(resume_file, "resumes")
            uploaded_resume_urls.append(url)
        
        return ResumeUploadResponse(resumes=uploaded_resume_urls)
        
    except Exception as e:
        # Clean up uploaded files if there's an error
        print(f"Error during resume upload: {str(e)}")
        
        # Extract S3 keys from URLs and delete files
        for url in uploaded_resume_urls:
            try:
                # Extract S3 key from URL
                blob_path = url.split(f"{azure_config.container_name}/")[-1]
                delete_file_from_azure(blob_path)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")
        
        raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

@app.post("/upload-job-descriptions", response_model=JobDescriptionUploadResponse)

async def upload_job_descriptions(
    job_description: UploadFile = File(..., description="Job description PDF file")
):
    """
    Upload a single job description PDF file to S3 bucket
    
    Args:
        job_description: Single job description PDF file
    
    Returns:
        JSON response with downloadable URL for uploaded job description file
    """
    
    # Validate that file is provided
    if not job_description:
        raise HTTPException(
            status_code=400, 
            detail="Job description file is required"
        )
    
    # Validate file type
    if not validate_pdf_file(job_description):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are allowed. Invalid file: {job_description.filename}"
        )
    
    # Check file size limits (5MB per file)
    max_file_size = 5 * 1024 * 1024  # 5MB
    if job_description.size and job_description.size > max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds 5MB limit: {job_description.filename}"
        )
    
    try:
        # Upload job description file
        await job_description.seek(0)
        url = await upload_file_to_azure(job_description, "job_descriptions")
        
        return JobDescriptionUploadResponse(job_descriptions=url)
        
    except Exception as e:
        # Clean up uploaded file if there's an error
        print(f"Error during job description upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job description upload failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Resume and Job Description Upload API with AI Matching",
        "version": "1.0.0",
        "endpoints": {
            "upload-resumes": "/upload-resumes - POST - Upload resume PDF files",
            "upload-job-descriptions": "/upload-job-descriptions - POST - Upload a single job description PDF file",
            "match": "/match/ - POST - Match resume with job description",
            "screen-candidates-from-urls": "/screen-candidates-from-urls/ - POST - Screen candidates from S3 URLs",
            "health": "/health - GET - Health check",
            "docs": "/docs - GET - API documentation"
        }
    }
