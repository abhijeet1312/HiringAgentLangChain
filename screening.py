import os
import re
import tempfile
import traceback
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()

from jai import send_bulk_email 

import json
from huggingface_hub import InferenceClient
from langchain.output_parsers import PydanticOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field, field_validator

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, Optional, List
from pydantic import Field


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import pandas as pd
from openai import AzureOpenAI
load_dotenv()
# api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# print(api_token)

from getpass import getpass

# HUGGINGFACEHUB_API_TOKEN = getpass()
import os

class CandidateAssessment(BaseModel):
    """Model for structured output of candidate assessment"""
    candidate_name: str = Field(description="The name of the candidate")
    skills_match_score: int = Field(description="Score from 1-10 on how well the candidate's skills match the requirements")
    experience_relevance_score: int = Field(description="Score from 1-10 on the relevance of candidate's experience")
    education_match_score: int = Field(description="Score from 1-10 on educational qualification match")
    overall_fit_score: int = Field(description="Score from 1-10 on overall fitness for the role")
    strengths: List[str] = Field(description="List of candidate's key strengths")
    weaknesses: List[str] = Field(description="List of candidate's key weaknesses")
    recommendation: str = Field(description="Short recommendation: 'Strong Match', 'Potential Match', or 'Not Recommended'")
    candidate_email:str = Field("The email of the candidate")
    candidate_phone: str = Field(default=None, description="The phone number of the candidate, if available")
    
    @field_validator('skills_match_score', 'experience_relevance_score', 'education_match_score', 'overall_fit_score')
    def score_must_be_valid(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('Score must be between 1 and 10')
        return v
    
    @field_validator('recommendation')
    def recommendation_must_be_valid(cls, v):
        valid_recommendations = ["Strong Match", "Potential Match", "Not Recommended"]
        if v not in valid_recommendations:
            raise ValueError(f'Recommendation must be one of: {", ".join(valid_recommendations)}')
        return v

class AzureOpenAILLM(LLM):
    client: Any = Field(...)
    deployment_id: str = Field(...)

    @property
    def _llm_type(self) -> str:
        return "azure_openai"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Azure OpenAI API error: {e}")
            return "Error: Unable to generate response"

class CandidateScreeningAgent:
    """Agent for screening job candidates using local LLMs"""
    
   

    def __init__(self, job_description: str):
        """
        Initialize the screening agent using Azure OpenAI .

        Args:
            job_description: The job description to screen candidates against
        """
        self.job_description = job_description

        
        self.gpt_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_CHAT")   
        )
        self.chat_deployment_id = "gpt-4.1-2"
        self.azure_llm = AzureOpenAILLM(
        client=self.gpt_client,
        deployment_id=self.chat_deployment_id
        )
        self.output_parser = PydanticOutputParser(pydantic_object=CandidateAssessment)
         

    
    def extract_phone_number(self, text: str) -> Optional[str]:
       """Extract phone number from text"""
       phone_pattern = r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
       phone_matches = re.findall(phone_pattern, text)
       return phone_matches[0] if phone_matches else None
    def load_resume(self, file_path: str) -> str:
        """
        Load and extract text from a resume file
        
        Args:
            file_path: Path to the resume file (PDF, DOCX, or TXT)
            
        Returns:
            Extracted text from the resume
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        
        elif file_extension == '.docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        
        elif file_extension == '.txt':
            loader = TextLoader(file_path)
            documents = loader.load()
            return "\n".join([doc.page_content for doc in documents])
        
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    
    
    
    

    def create_assessment_prompt(self, resume_text: str) -> PromptTemplate:
        """Create the prompt template for candidate assessment"""
    
        format_instructions = self.output_parser.get_format_instructions()
        template="""
You are an expert HR recruiter. Analyze the candidate's resume against the job description below.

Return ONLY a valid JSON object that strictly follows the schema. Do NOT include explanations or formatting like code blocks. The response MUST be a plain JSON object.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Schema:
{{
  "candidate_name": "string",
  "skills_match_score": integer (1-10),
  "experience_relevance_score": integer (1-10),
  "education_match_score": integer (1-10),
  "overall_fit_score": integer (1-10),
  "strengths": [string],
  "weaknesses": [string],
  "recommendation": "Strong Match" | "Potential Match" | "Not Recommended",
  "candidate_email": "string"
  "candidate_phone": "string"  # Optional, can be null
}}
"""

       
    
        return PromptTemplate(
        input_variables=["job_description", "resume_text"],
        partial_variables={"format_instructions": format_instructions},
        template=template
    )
        
   
    def screen_candidate(self, resume_path: str) -> CandidateAssessment:
        """
        Screen a candidate's resume against the job description
        
        Args:
            resume_path: Path to the candidate's resume file
            
        Returns:
            Structured assessment of the candidate
        """
        # Load and extract text from resume
        resume_text = self.load_resume(resume_path)
        
        # Create prompt for assessment
        prompt = self.create_assessment_prompt(resume_text)
        
        #build new chain
        
        chain= prompt | self.azure_llm | self.output_parser 
        result = None
        try:
            result =chain.invoke({
                "job_description":self.job_description,
                "resume_text":resume_text
            })
            return result
        except Exception as e:
            print(e)
            print(f"\nError Parsing output:\n{e}\n")
            print("=== RAW LLM OUTPUT START ===")
            print(result)
            print("=== RAW LLM OUTPUT END ===")
            return None
        
    def batch_screen_candidates(self, resume_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Screen multiple candidates and return assessments
        
        Args:
            resume_paths: List of paths to candidate resume files
            
        Returns:
            List of candidate assessments
        """
        results = []
        
        for path in resume_paths:
            candidate_name = os.path.basename(path).split('.')[0]
            print(f"Screening candidate: {candidate_name}")
            
            try:
                assessment = self.screen_candidate(path)
                
                if isinstance(assessment, CandidateAssessment):
                    results.append(assessment.dict())
                else:
                    # Handle case where parsing failed
                    results.append({
                        "candidate_name": candidate_name,
                        "raw_assessment": assessment,
                        "error": "Failed to parse structured output"
                    })
            except Exception as e:
                print(f"Error screening candidate {candidate_name}: {e}")
                traceback.print_exc()
                results.append({
                    "candidate_name": candidate_name,
                    "error": str(e)
                })
        
        return results
    
 

    def generate_report(self, assessments: List[Dict[str, Any]], output_path: str = "candidate_assessments.csv",voice_interview_threshold: float = 3.0):
        """
    Generate a CSV report from candidate assessments, including KMeans-based PASS/FAIL status.
    
    Args:
        assessments: List of candidate assessments
         output_path: Path to save the CSV report
        """
        df = pd.DataFrame(assessments)

    # Apply KMeans clustering if 'overall_fit_score' exists
        if 'overall_fit_score' in df.columns:
          scores = df['overall_fit_score'].values.reshape(-1, 1)
          kmeans = KMeans(n_clusters=1, random_state=42)
          df['cluster'] = kmeans.fit_predict(scores)

        # Identify top-performing cluster based on centroid value
          centroids = kmeans.cluster_centers_.flatten()
          top_cluster = np.argmax(centroids)  # cluster with highest average score

        # Assign PASS/FAIL based on cluster membership
          df['status'] = df['cluster'].apply(lambda x: 'PASS' if x == top_cluster else 'FAIL')

    # Save report to CSV
        df.to_csv(output_path, index=False)
        print(f"Report generated and saved to {output_path}")

               
        if 'overall_fit_score' in df.columns:
           top_candidates = df.sort_values(by='overall_fit_score', ascending=False).head(5)
           print("\nTop 5 Candidates:")
           display_cols = ['candidate_name', 'overall_fit_score', 'recommendation', 'candidate_email']
           if 'status' in df.columns:
             display_cols.append('status')
    
           print(top_candidates[display_cols])

           # Initialize return data
           qualified_candidates_for_voice = []
           email_recipients = []
           if (top_candidates['overall_fit_score'] >= 3.0).any():
            high_scorers = top_candidates[top_candidates['overall_fit_score'] >=3.0]
            print("Candidates with score > 6:")
            print(high_scorers[['candidate_name','overall_fit_score' ,'candidate_email']])
        
         # Extract list of emails from high_scorers
            email_recipients  = high_scorers['candidate_email'].tolist()
            print(len(email_recipients))
            # Prepare candidate data for voice interviews
            for index, row in high_scorers.iterrows():
                candidate_data = {
                    "id": index + 1,  # Assign unique ID
                    "name": row['candidate_name'],
                    "email": row['candidate_email'],
                    # "phone": self.extract_phone_number(row.get('candidate_email', '')),  # You'll need to implement this
                    "phone": row.get('candidate_phone', None),
                    # "phone":"+918887596182",
                    "resume_score": row['overall_fit_score'],
                    "recommendation": row.get('recommendation', ''),
                    "strengths": row.get('strengths', []),
                    "weaknesses": row.get('weaknesses', []),
                    "status": row.get('status', 'UNKNOWN')
                }
                qualified_candidates_for_voice.append(candidate_data)

            
            

           if len(email_recipients) > 0:
               
            message = "Congratulations! You have been shortlisted based on your profile."
            # print(job_description)
            # receiver=["abhijeetsrivastava2189@gmail.com"]
            #job desc current stage next stage
            current_stage="Resume Screening Phase"
            next_stage="Voice Interview Round"
            email_recipients.append("abhijeetsrivastava2189@gmail.com")
            print(f"Sending emails to: {email_recipients}")
            # receiver.append("Aurjobsa@gmail.com")
            print(email_recipients)
            
            try:
                    send_bulk_email(email_recipients, self.job_description, current_stage, next_stage)
                    print("✅ Emails sent successfully")
            except Exception as e:
                    print(f"❌ Error sending emails: {e}")
      # Return structured data for voice interviews
        return {
        "qualified_candidates": qualified_candidates_for_voice,
        "total_qualified": len(qualified_candidates_for_voice),
        "email_recipients": email_recipients,
        "threshold_used": voice_interview_threshold
    }         
             
    # Function to trigger voice interviews using the returned data
    def trigger_voice_interviews_for_qualified(self,qualified_data: Dict):
        """
        Trigger voice interviews for qualified candidates
    
        Args:
        qualified_data: Data returned from generate_report
        job_description: Job description for the position
         """
        if not qualified_data['qualified_candidates']:
           print("No qualified candidates for voice interview")
           return
        print(f"\n🎤 Starting voice interviews for {qualified_data['total_qualified']} candidates")
       
        # Initialize voice interview agent
        from langchain_prescreening_agent import create_prescreening_agent
        voice_agent = create_prescreening_agent()
    
    # Prepare input for voice interviewer
        voice_input = {
        "candidates": qualified_data['qualified_candidates'],
        "job_description": self.job_description
         }
        # print(voice_input)
    
        # Run voice interviews
        print("📞 Initiating voice interviews...")
        voice_results = voice_agent.run_pre_screening(json.dumps(voice_input))
    
        return voice_results
           
def extract_key_info_from_resume(resume_text: str) -> Dict[str, Any]:
    """
    Extract key information from a resume
    
    Args:
        resume_text: Text content of the resume
        
    Returns:
        Dictionary containing extracted information
    """
    info = {
        "name": None,
        "email": None,
        "phone": None,
        "education": [],
        "experience": [],
        "skills": []
    }
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_matches = re.findall(email_pattern, resume_text)
    if email_matches:
        info["email"] = email_matches[0]
    
    # Extract phone
    phone_pattern = r'\b(?:\+\d{1,3}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    phone_matches = re.findall(phone_pattern, resume_text)
    if phone_matches:
        info["phone"] = phone_matches[0]
    
    # Education section detection
    education_section = re.search(r'(?i)education.*?(?=experience|skills|$)', resume_text, re.DOTALL)
    if education_section:
        edu_text = education_section.group(0)
        # Look for degree patterns
        degree_patterns = [
            r'\b(?:B\.?S\.?|Bachelor of Science|Bachelor\'s)\b.*?(?:\d{4}|\d{2})',
            r'\b(?:M\.?S\.?|Master of Science|Master\'s)\b.*?(?:\d{4}|\d{2})',
            r'\b(?:Ph\.?D\.?|Doctor of Philosophy|Doctorate)\b.*?(?:\d{4}|\d{2})'
        ]
        
        for pattern in degree_patterns:
            matches = re.findall(pattern, edu_text)
            info["education"].extend(matches)
    
    # Skills section detection (simple approach)
    skills_section = re.search(r'(?i)skills.*?(?=experience|education|$)', resume_text, re.DOTALL)
    if skills_section:
        skills_text = skills_section.group(0)
        # Extract words that might be skills
        potential_skills = re.findall(r'\b[A-Za-z+#\.]+\b', skills_text)
        # Filter out common words
        common_words = {'skills', 'and', 'the', 'with', 'in', 'of', 'a', 'to', 'for'}
        info["skills"] = [s for s in potential_skills if len(s) > 2 and s.lower() not in common_words]
    
    return info
if __name__ == "__main__":
    # Example job description
    job_description = """
    Software Engineer - Full Stack Developer
    
    Requirements:
    - 3+ years of experience in web development
    - Proficiency in Python, JavaScript, and React
    - Experience with databases (SQL/NoSQL)
    - Knowledge of cloud platforms (AWS, Azure, or GCP)
    - Strong problem-solving skills
    - Bachelor's degree in Computer Science or related field
    
    Responsibilities:
    - Develop and maintain web applications
    - Collaborate with cross-functional teams
    - Write clean, maintainable code
    - Participate in code reviews
    """
    
    # Initialize the screening agent
    print("🚀 Initializing Candidate Screening Agent...")
    screening_agent = CandidateScreeningAgent(job_description)
    
    # Example resume paths (you can modify these)
    resume_folder = "resumes"  # Create this folder and put resume files
    
    # Check if resume folder exists
    if not os.path.exists(resume_folder):
        print(f"❌ Resume folder '{resume_folder}' not found!")
        print("Please create a 'resumes' folder and add resume files (.pdf, .docx, .txt)")
        
        # Create example with single resume file
        print("\n📄 Testing with single resume file...")
        resume_file = input("Enter path to a resume file (or press Enter to skip): ").strip()
        
        if resume_file and os.path.exists(resume_file):
            print(f"Screening single candidate: {resume_file}")
            try:
                assessment = screening_agent.screen_candidate(resume_file)
                if assessment:
                    print(f"\n✅ Assessment Result:")
                    print(f"Name: {assessment.candidate_name}")
                    print(f"Overall Score: {assessment.overall_fit_score}/10")
                    print(f"Recommendation: {assessment.recommendation}")
                    print(f"Strengths: {', '.join(assessment.strengths)}")
                    print(f"Weaknesses: {', '.join(assessment.weaknesses)}")
                else:
                    print("❌ Failed to assess candidate")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("No valid resume file provided. Exiting.")
            exit()
    else:
        # Get all resume files from the folder
        resume_files = []
        for file in os.listdir(resume_folder):
            if file.lower().endswith(('.pdf', '.docx', '.txt')):
                resume_files.append(os.path.join(resume_folder, file))
        
        if not resume_files:
            print(f"❌ No resume files found in '{resume_folder}' folder!")
            print("Please add resume files with extensions: .pdf, .docx, .txt")
            exit()
        
        print(f"📋 Found {len(resume_files)} resume files")
        for i, file in enumerate(resume_files, 1):
            print(f"  {i}. {os.path.basename(file)}")
        
        # Batch screen all candidates
        print("\n🔍 Starting batch screening...")
        assessments = screening_agent.batch_screen_candidates(resume_files)
        
        # Generate report
        print("\n📊 Generating report...")
        qualified_data = screening_agent.generate_report(assessments)
        
        # Display results
        print(f"\n✅ Screening completed!")
        print(f"Total candidates screened: {len(assessments)}")
        print(f"Qualified for voice interview: {qualified_data['total_qualified']}")
        
        if qualified_data['total_qualified'] > 0:
            print("\n🎤 Qualified candidates:")
            for candidate in qualified_data['qualified_candidates']:
                print(f"  - {candidate['name']} (Score: {candidate['resume_score']}/10)")
            
            print("\nEmails sent to qualified candidates.")
        
        print(f"\n📄 Report saved to: candidate_assessments.csv")
        print("🎯 Screening process completed!")