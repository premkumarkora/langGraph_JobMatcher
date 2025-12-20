import streamlit as st
import os
import shutil
import base64
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import json
from typing import Dict, TypedDict, List
from job_matcher import JobState, extract_skills_node, search_jobs_node, score_jobs_node, select_top_jobs_node

# Load env variables
load_dotenv()

# Constants
DB_PATH = "/Volumes/vibecoding/landchainDemo/chroma_db"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_NAME = "gpt-4o-mini"

# --- Streamlit Config ---
st.set_page_config(page_title="Job Matcher Agent", layout="wide")

# Load Personal Info
@st.cache_resource
def load_personal_info():
    try:
        with open("/Volumes/vibecoding/landchainDemo/info.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load info.json: {e}")
        return {}

personal_info = load_personal_info()

# --- Helpers ---
def get_mermaid_url(mermaid_code: str) -> str:
    """
    Converts mermaid code to a mermaid.ink URL for robust web rendering.
    """
    json_obj = {"code": mermaid_code, "mermaid": {"theme": "dark"}}
    json_str = json.dumps(json_obj)
    base64_str = base64.b64encode(json_str.encode("utf-8")).decode("utf-8")
    return f"https://mermaid.ink/img/{base64_str}"

# --- Ingestion Logic ---
def ingest_file(uploaded_file):
    """
    Clears existing DB, saves file, and ingests into Chroma.
    """
    # 1. Clear existing DB
    if os.path.exists(DB_PATH):
        try:
            import time
            shutil.rmtree(DB_PATH)
            # Give the OS a moment to release file handles
            time.sleep(1)
            st.success(f"Cleared existing database at {DB_PATH}")
        except Exception as e:
            st.error(f"Error deleting DB: {e}")
            return False

    # 2. Save uploaded file to temp path
    save_path = f"/Volumes/vibecoding/landchainDemo/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"File saved to {save_path}. Starting ingestion...")

    # 3. Ingest
    try:
        loader = Docx2txtLoader(save_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=DB_PATH
        )
        st.success(f"Successfully ingested {len(splits)} chunks into ChromaDB!")
        return True
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        return False

# --- Graph Logic ---
# We use @st.cache_resource to ensure the 'memory' checkpointer persists across Streamlit reruns.
@st.cache_resource
def get_app():
    workflow = StateGraph(JobState)
    workflow.add_node("extract_skills", extract_skills_node)
    workflow.add_node("search_jobs", search_jobs_node)
    workflow.add_node("score_jobs", score_jobs_node)
    workflow.add_node("select_top_jobs", select_top_jobs_node)

    workflow.set_entry_point("extract_skills")
    workflow.add_edge("extract_skills", "search_jobs")
    workflow.add_edge("search_jobs", "score_jobs")
    workflow.add_edge("score_jobs", "select_top_jobs")
    workflow.add_edge("select_top_jobs", END)

    # Compile with checkpointer
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["search_jobs"])

app = get_app()

def main():
    st.title("🤖 AI Job Matcher Agent")
    st.markdown("Upload your CV to find matched jobs using LangGraph.")

    # Sidebar for file upload
    with st.sidebar:
        st.header("1. Upload CV")
        uploaded_file = st.file_uploader("Choose a DOCX file", type=["docx", "doc"])
        
        if uploaded_file is not None:
            if st.button("Ingest CV"):
                with st.spinner("Ingesting..."):
                     if ingest_file(uploaded_file):
                         st.session_state["ingested"] = True
        
        if st.session_state.get("ingested"):
             st.success("CV Ready for Analysis")

    # Main Area
    if not st.session_state.get("ingested"):
        st.info("Please upload and ingest a CV to get started.")
        return

    # Initialize Graph Config
    thread_id = "streamlit_session"
    config = {"configurable": {"thread_id": thread_id}}

    # Start Button
    if st.button("Start Analysis"):
        st.session_state["analyzing"] = True
        # Initial run triggers extraction
        st.info("Starting Workflow...")
        
        # Invoke with a dummy config just to verify we can step
        # We use `app.invoke` if we want to run until interrupt, but `stream` is safer for UI feedback
        # For Streamlit, we want to run up to the interrupt point.
        # app.stream yields steps. We consume them.
        try:
            for event in app.stream({"job_title": "", "skills": "", "job_listings": [], "scored_jobs": [], "final_output": ""}, config):
                # event key is the node name, value is the state update
                for node_name, state_update in event.items():
                    if node_name == "extract_skills":
                        title = state_update.get("job_title", "Unknown")
                        st.success(f"✅ Skills Extracted. Inferred Title: **{title}**")
                    elif node_name == "search_jobs":
                        count = len(state_update.get("job_listings", []))
                        st.success(f"✅ Search Completed. Found {count} listings.")
                    elif node_name == "score_jobs":
                        st.success("✅ Jobs Scored by LLM.")
                    elif node_name == "select_top_jobs":
                        st.success("✅ Top Candidates Selected.")
                    else:
                        st.write(f"Executed node: {node_name}")
        except Exception as e:
             st.error(f"Error during execution: {e}")
        st.rerun()

    # Handling State
    current_state = app.get_state(config)
    
    # Render Graph if analysis has started
    if st.session_state.get("analyzing"):
        st.divider()
        st.subheader("📊 Workflow Progress")
        # try:
        #     mermaid_code = app.get_graph().draw_mermaid()
        #     url = get_mermaid_url(mermaid_code)
        #     st.image(url)
        # except Exception as e:
        #     st.warning("Visual graph preview unavailable.")
        #     st.code(app.get_graph().draw_mermaid(), language="mermaid")
    
    # If state exists and we are analyzing
    if current_state.values:
        st.divider()
        st.write("### Current Pipeline State")
        
        # Display Extraction Results
        job_title = current_state.values.get("job_title")
        skills = current_state.values.get("skills")
        
        if job_title and skills:
             st.info(f"**Inferred Job Title:** {job_title}")
             with st.expander("Extracted Skills"):
                 st.write(skills)
                 
        # HITL Check: If we are interrupted before search_jobs
        if current_state.next and "search_jobs" in current_state.next:
            st.warning("⚠️ Human Approval Required")
            st.write("The Agent inferred the Job Title above. You can approve or edit it before searching.")
            
            new_title = st.text_input("Refine Job Title", value=job_title)
            
            if st.button("Approve & Search"):
                # Update state with new title if changed
                app.update_state(config, {"job_title": new_title})
                
                # Resume execution
                st.info(f"🔄 Resuming with Job Title: **{new_title}**...")
                try:
                    # Continue stream
                    for event in app.stream(None, config):
                        for node_name, state_update in event.items():
                            if node_name == "search_jobs":
                                count = len(state_update.get("job_listings", []))
                                st.success(f"✅ Search Completed. Found {count} listings.")
                            elif node_name == "score_jobs":
                                st.success("✅ Jobs Scored by LLM.")
                            elif node_name == "select_top_jobs":
                                st.success("✅ Top Candidates Selected.")
                except Exception as e:
                    st.error(f"Error resuming: {e}")
                st.rerun()

        # Display Final Results if completed
        final_output = current_state.values.get("final_output")
        scored_jobs = current_state.values.get("scored_jobs", [])
        
        if final_output:
            st.divider()
            st.success(" Analysis Complete!")
            
            # Show all listings (No expander)
            if scored_jobs:
                st.subheader(f"🔍 All Found Listings ({len(scored_jobs)})")
                for i, job in enumerate(scored_jobs, 1):
                    score = job.get('match_score', 0)
                    st.markdown(f"**{i}. {job.get('title')}** at *{job.get('company_name')}*")
                    st.markdown(f"**Confidence Score:** <span style='color:green; font-weight:bold;'>{score}/100</span>", unsafe_allow_html=True)
                    st.caption(f"Location: {job.get('location')} | Via: {job.get('via')}")
                    if job.get('description'):
                        with st.expander("View Description"):
                            st.text(job.get('description')[:500] + "...")
                    st.divider()

            st.write("### Top Recommended Matches")
            
            # Helper to generate bundle
            def handle_bundle_generation(job):
                with st.spinner(f"Generating bundle for {job.get('title')}..."):
                    # 1. Fetch full CV from Chroma
                    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
                    results = vectorstore.similarity_search("full experience history project details education", k=5)
                    full_cv = "\n".join([doc.page_content for doc in results])
                    
                    # 2. Create Dir
                    dirname = f"{job.get('title')}_{job.get('company_name')}_{job.get('location')}".replace(" ", "_").replace("/", "_")
                    os.makedirs(dirname, exist_ok=True)
                    
                    # 3. Generate Contents
                    from job_matcher import generate_tailored_resume, generate_cover_letter
                    from docx_utils import create_professional_docx, create_job_info_docx
                    
                    resume_text = generate_tailored_resume(full_cv, job.get('description', ''), personal_info=personal_info)
                    cover_text = generate_cover_letter(full_cv, job, personal_info=personal_info)
                    
                    # 4. Save Files
                    create_professional_docx(os.path.join(dirname, "Tailored_Resume.docx"), f"Resume: {job.get('title')}", resume_text)
                    create_professional_docx(os.path.join(dirname, "Cover_Letter.docx"), f"Cover Letter: {job.get('company_name')}", cover_text)
                    create_job_info_docx(os.path.join(dirname, "Job_Reference.docx"), job)
                    
                    st.success(f"✅ Bundle created in folder: `{dirname}`")

            # Display top 3 with buttons
            top_3 = scored_jobs[:3]
            for i, job in enumerate(top_3, 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {i}. {job.get('title')}")
                    st.markdown(f"**Company:** {job.get('company_name')} | **Location:** {job.get('location')}")
                    st.markdown(f"**Confidence Score:** <span style='color:green; font-weight:bold;'>{job.get('match_score')}/100</span>", unsafe_allow_html=True)
                    st.write(f"**Reason:** {job.get('match_reason')}")
                with col2:
                    if st.button(f"📄 Generate .docx Bundle", key=f"btn_{i}"):
                        handle_bundle_generation(job)
                st.divider()

if __name__ == "__main__":
    main()
