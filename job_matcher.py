import os
import json
from typing import Dict, TypedDict, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from serpapi import GoogleSearch

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load env variables
load_dotenv()

# Constants
DB_PATH = "/Volumes/vibecoding/landchainDemo/chroma_db"
# Mapping "gpt-4.1-mini" request to "gpt-4o-mini"
MODEL_NAME = "gpt-4o-mini" 

# --- State Definition ---
# --- State Definition ---
class JobState(TypedDict):
    """
    Represents the state of the job matching workflow.
    
    Attributes:
        job_title (str): The inferred or user-approved job title to use for searching.
        skills (str): A summary of the candidate's key technical skills extracted from the CV.
        job_listings (List[Dict]): A list of raw job dictionaries returned by the Google Search API.
        scored_jobs (List[Dict]): The same job listings, enriched with 'match_score' (0-100) and 'match_reason'.
        final_output (str): The formatted string containing the top recommended jobs for display.
    """
    job_title: str
    skills: str
    job_listings: List[Dict]
    scored_jobs: List[Dict]
    final_output: str

# --- Helper Functions ---

def find_jobs(query_text, num_results=10):
    """
    Searches for jobs using SerpApi's Google Jobs engine.
    
    This helper function abstracts the interaction with the external search API.
    It handles authentication via the SERP_API environment variable and structures
    the request for the 'google_jobs' engine.

    Args:
        query_text (str): The search query string (e.g., "Generative AI Engineer jobs").
        num_results (int): The maximum number of job listings to return (default: 10).

    Returns:
        List[Dict]: A list of dictionaries, where each dictionary represents a job listing
                    containing fields like 'title', 'company_name', 'location', 'description', etc.
                    Returns an empty list if the API key is missing or an error occurs.
    """
    api_key = os.getenv("SERP_API")
    if not api_key:
        print("Error: SERP_API environment variable not set.")
        return []

    print(f"Searching Google for: '{query_text}'...")

    # Configure search parameters for SerpApi
    # engine="google_jobs" targets the specialized job search widget on Google
    params = {
        "engine": "google_jobs",
        "q": query_text,
        "api_key": api_key,
        "google_domain": "google.com",
        "gl": "us",     # Geo-location: United States
        "hl": "en"      # Language: English
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # for google_jobs engine, the key is usually 'jobs_results'
        jobs_results = results.get("jobs_results", [])
        
        # Return only the requested number of results to manage token usage later
        return jobs_results[:num_results]

    except Exception as e:
        print(f"An error occurred during the search: {e}")
        return []

# --- Node Functions ---

def extract_skills_node(state: JobState):
    """
    1. Fetch CV content from ChromaDB.
    2. Use LLM to identify the best matching Job Title and a summary of skills.
    """
    print("--- Node: Extract Skills & Title ---")
    
    # Connect to Chroma
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_function
    )
    
    # Query for skills
    results = vectorstore.similarity_search("technical skills programming languages tools frameworks", k=3)
    cv_text = "\n".join([doc.page_content for doc in results])
    
    # LLM extraction
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    prompt = f"""
    You are an expert career coach.
    Analyze the following CV segments.
    
    1. Identify the SINGLE most appropriate specific Job Title for this candidate (e.g. "AI Engineer", "Senior Python Backend Developer").
    2. Extract a summary of their key technical skills (to used for scoring job relevance later).
    
    CV Segments:
    {cv_text}
    
    Return the result as a valid JSON object:
    {{
        "job_title": "The inferred job title",
        "skills": "Comma separated list of key skills"
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip().replace('```json', '').replace('```', '')
        data = json.loads(content)
        
        job_title = data.get("job_title", "Software Engineer")
        skills = data.get("skills", "")
        
        print(f"Inferred Job Title: {job_title}")
        print(f"Extracted Skills: {skills}")
        
        return {"job_title": job_title, "skills": skills}
        
    except Exception as e:
        print(f"Error extracting skills/title: {e}")
        return {"job_title": "Software Engineer", "skills": "Python"}

def approval_node(state: JobState):
    """
    NODE: Human Approval (HITL)
    
    This node implements a Human-in-the-Loop checkpoint. It pauses the automated workflow
    to present the inferred "Job Title" to the user. The user can either approve the
    LLM's inference or override it with their own preferred search term.
    
    Args:
        state (JobState): Current process state containing the inferred 'job_title'.
        
    Returns:
        Dict: Updates 'job_title' in the state based on user input.
    """
    print("--- Node: Human Approval ---")
    current_title = state["job_title"]
    
    print(f"\n--- Approval Required ---")
    print(f"Inferred Job Title: {current_title}")
    # Show a snippet of skills to give context on why this title was chosen
    print(f"Extracted Skills: {state['skills'][:100]}...")
    
    print(f"Press 'Enter' to approve, or type a new Job Title to override.")
    
    # Blocking call to get user input from the console
    user_input = input(" > ")
    
    if user_input.strip():
        # User provided an override
        print(f"Updating Job Title to: {user_input}")
        return {"job_title": user_input}
    
    # User approved the existing title
    print("Job Title approved.")
    return {"job_title": current_title}

def search_jobs_node(state: JobState):
    """
    Use JobSearchTool to find jobs based on the inferred Job Title.
    """
    print("--- Node: Search Jobs ---")
    job_title = state["job_title"]
    
    # Create a search query using the title
    # We can append 'jobs' or 'remote' if desired
    query = f"{job_title} jobs"
    print(f"Searching for: {query}")
    
    jobs = find_jobs(query, num_results=10)
    print(f"Found {len(jobs)} jobs.")
    
    return {"job_listings": jobs}

def score_jobs_node(state: JobState):
    """
    NODE: Score Jobs
    
    This node acts as a "AI Recruiter". It takes the list of raw job listings found
    by the search engine and evaluates them one by one against the candidate's skills.
    
    It uses the LLM to assign a relevance score (0-100) and provide a reasoning.
    
    Args:
        state (JobState): Contains 'skills', 'job_title', and 'job_listings'.
        
    Returns:
        Dict: Updates 'scored_jobs' in the state with the list of scored/ranked jobs.
    """
    print("--- Node: Score Jobs ---")
    skills = state["skills"]
    job_title = state["job_title"]
    jobs = state["job_listings"]
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    
    scored_results = []
    
    if not jobs:
        print("No jobs to score.")
        return {"scored_jobs": []}
    
    print(f"Scoring {len(jobs)} jobs against profile...")
    
    for job in jobs:
        # Create a concise description for the LLM to avoid token limits
        job_desc = f"Title: {job.get('title')}\nCompany: {job.get('company_name')}\nSnippet: {job.get('description', '')}"
        
        # Prompt Engineering:
        # We instruct the model to act as a recruiter and output strictly formatted JSON.
        prompt = f"""
        You are a recruiter matching candidates to jobs.
        Candidate Job Target: {job_title}
        Candidate Skills: {skills}
        
        Job Description:
        {job_desc}
        
        Task:
        1. Score this job match from 0 to 100 based on relevance.
        2. Provide a brief reason (1 sentence).
        
        Output format needs to be valid JSON:
        {{
            "score": 85,
            "reason": "Matches Python and AI skills."
        }}
        """
        
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            # Clean up potential markdown code blocks (```json ... ```) common in LLM responses
            content = response.content.replace("```json", "").replace("```", "").strip()
            score_data = json.loads(content)
            
            job_with_score = job.copy()
            job_with_score["match_score"] = score_data.get("score", 0)
            job_with_score["match_reason"] = score_data.get("reason", "")
            scored_results.append(job_with_score)
            
        except Exception as e:
            print(f"Error scoring job {job.get('title')}: {e}")
            # Fallback for failed scoring to ensure pipeline continues
            job_with_score = job.copy()
            job_with_score["match_score"] = 0
            job_with_score["match_reason"] = "Error scoring"
            scored_results.append(job_with_score)
            
    # Sort criteria: Highest score first
    scored_results.sort(key=lambda x: x["match_score"], reverse=True)
    
    return {"scored_jobs": scored_results}

def select_top_jobs_node(state: JobState):
    """
    Select top 3 and format detailed output.
    """
    print("--- Node: Select Top Jobs ---")
    top_jobs = state["scored_jobs"][:3]
    
    output_lines = ["# Top 3 Job Matches\n"]
    
    for i, job in enumerate(top_jobs, 1):
        line = f"""
## {i}. {job.get('title')}
**Company:** {job.get('company_name')}
**Location:** {job.get('location')}
**Confidence Score:** <span style="color:green">{job.get('match_score')}/100</span>
**Reason:** {job.get('match_reason')}
**Via:** {job.get('via')}
**Description Snippet:**
{job.get('description', 'No description available')}
---
"""
        output_lines.append(line)
        
    final_str = "\n".join(output_lines)
    print(final_str) # Print to console as requested
    
    return {"final_output": final_str}

# --- Graph Construction ---

workflow = StateGraph(JobState)

workflow.add_node("extract_skills", extract_skills_node)
workflow.add_node("approval", approval_node)
workflow.add_node("search_jobs", search_jobs_node)
workflow.add_node("score_jobs", score_jobs_node)
workflow.add_node("select_top_jobs", select_top_jobs_node)

workflow.set_entry_point("extract_skills")

workflow.add_edge("extract_skills", "approval")
workflow.add_edge("approval", "search_jobs")
workflow.add_edge("search_jobs", "score_jobs")
workflow.add_edge("score_jobs", "select_top_jobs")
workflow.add_edge("select_top_jobs", END)

app = workflow.compile()

def main():
    print("Starting Job Matcher Workflow...")
    final_state = app.invoke({
        "job_title": "",
        "skills": "",
        "job_listings": [],
        "scored_jobs": [],
        "final_output": ""
    })
    print("\nWorkflow Completed.")

def generate_tailored_resume(original_cv_text, job_description, personal_info=None):
    """
    Uses LLM to rewrite the CV text to emphasize skills matching the JD.
    Integrates personal info and avoids markdown.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
    personal_str = f"Name: {personal_info.get('name', '')}\nAddress: {personal_info.get('address', {}).get('full_address', '')}\nLinkedIn: {personal_info.get('social_links', {}).get('linkedin', '')}\nGitHub: {personal_info.get('social_links', {}).get('github', '')}" if personal_info else ""
    
    prompt = f"""
    You are an expert resume writer. 
    Rewrite the following resume to perfectly match this Job Description.
    
    USER PERSONAL INFO (Include this at the top):
    {personal_str}
    
    JOB DESCRIPTION:
    {job_description}
    
    ORIGINAL RESUME CONTENT:
    {original_cv_text}
    
    STRICT RULES:
    1. Output ONLY the resume text.
    2. DO NOT use ANY Markdown formatting (No asterisks like **, no hashes like #, no underscores like __).
    3. Use a clear layout with section headers in ALL CAPS.
    4. Use plain text bullet points (e.g., "- " at the start of a line).
    5. Ensure the candidate's personal contact info provided above is at the very top.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_cover_letter(original_cv_text, job_data, personal_info=None):
    """
    Uses LLM to generate a professional cover letter.
    Integrates personal info and avoids markdown.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)
    job_desc = job_data.get('description', '')
    company = job_data.get('company_name', 'Hiring Manager')
    role = job_data.get('title', 'Target Role')
    
    personal_str = f"Name: {personal_info.get('name', '')}\nAddress: {personal_info.get('address', {}).get('full_address', '')}\nLinkedIn: {personal_info.get('social_links', {}).get('linkedin', '')}\nGitHub: {personal_info.get('social_links', {}).get('github', '')}" if personal_info else ""

    prompt = f"""
    Write a professional and compelling one-page business cover letter for the role of {role} at {company}.
    
    CANDIDATE CONTACT INFO (Use this for the header):
    {personal_str}
    
    JOB DESCRIPTION SNIPPET:
    {job_desc[:2000]}
    
    CANDIDATE BACKGROUND:
    {original_cv_text[:2000]}
    
    STRICT RULES:
    1. Output ONLY the cover letter text.
    2. DO NOT use ANY Markdown formatting (No asterisks **, no hashes #).
    3. Use a formal business letter structure.
    4. Sign off with the candidate's name: {personal_info.get('name', 'PremKumar Kora')}.
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

if __name__ == "__main__":
    main()
