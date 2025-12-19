# Job Matcher Application Requirements Document

**Version:** 1.0
**Date:** 2025-12-17
**Status:** Approved

## 1. Introduction
The Job Matcher Application is an automated tool designed to assist job seekers by matching their Curriculum Vitae (CV) stored in a vector database with relevant live job listings. It leverages Large Language Models (LLMs) and Google Search to identify, filter, and score job opportunities based on the candidate's specific skills and experience.

## 2. Problem Statement
Manual job searching is time-consuming and inefficient. Candidates often struggle to identify the most relevant roles among thousands of listings. Typical keyword searches fail to account for the depth of a candidate's specific technical expertise and context.

## 3. Goals and Objectives
*   **Automate Job Discovery**: Automatically find jobs relevant to the candidate's profile.
*   **Context-Aware Matching**: Use LLMs to understand the semantic meaning of skills rather than simple keyword matching.
*   **Human-in-the-Loop (HITL)**: Allow the user to verify and approve the search strategy before execution.
*   **Scoring & Ranking**: Provide a clear relevance score (0-100) and reasoning for every job match.

## 4. Functional Requirements

### 4.1. Data Ingestion & Storage
*   **FR-01**: System MUST be able to query a local Chroma Vector Database containing the candidate's CV embeddings.
*   **FR-02**: System MUST retrieve relevant CV segments using similarity search for keywords like "technical skills" and "experience".

### 4.2. Intelligence & Analysis
*   **FR-03**: System MUST use an LLM (specifically `gpt-4o-mini`) to analyze CV segments.
*   **FR-04**: System MUST infer a single, specific "Job Title" (e.g., "Generative AI Engineer") best suited for the candidate.
*   **FR-05**: System MUST extract a summary list of key technical skills.

### 4.3. User Interaction (HITL)
*   **FR-06**: System MUST pause execution after inferring the Job Title.
*   **FR-07**: System MUST display the inferred title to the user via the command line interface (CLI).
*   **FR-08**: System MUST allow the user to approve the inferred title (by pressing Enter) or override it with a custom title.

### 4.4. Job Search API
*   **FR-09**: System MUST use SerpApi (Google Jobs Engine) to search for live job listings.
*   **FR-10**: System MUST use the approved Job Title as the primary search query.
*   **FR-11**: System MUST retrieve a minimum of 10 job listings including metadata (Company, Location, Description Snippet).

### 4.5. Scoring & ranking
*   **FR-12**: System MUST use an LLM to compare each job's description against the candidate's extracted skills.
*   **FR-13**: System MUST generate a relevance score (0-100) for each job.
*   **FR-14**: System MUST provide a brief text justification for the score.
*   **FR-15**: System MUST sort the final results in descending order of their match score.

### 4.6. Output
*   **FR-16**: System MUST display the Top 3 scored jobs in a readable format.
*   **FR-17**: Output MUST include Title, Company, Location, Score, Reason, and a Description Snippet.

## 5. Non-Functional Requirements

### 5.1. Performance
*   **NFR-01**: Analysis and search workflow should utilize agentic orchestration (LangGraph) for efficient state management.
*   **NFR-02**: Orchestration should minimize latency between steps, contingent on external API response times.

### 5.2. Security & configuration
*   **NFR-03**: API Keys (SerpApi, OpenAI) MUST be stored securely in environment variables (`.env`).
*   **NFR-04**: No sensitive CV data should be exposed externally other than to the LLM and the local console.

### 5.3. Reliability
*   **NFR-05**: System should handle API failures (e.g., scoring errors) gracefully by returning a 0 score rather than crashing.
*   **NFR-06**: Deprecation warnings from underlying libraries (LangChain) should be suppressed or resolved for clean output.

## 6. System Architecture Summary
*   **Orchestrator**: LangGraph (StateGraph)
*   **LLM Provider**: OpenAI (`gpt-4o-mini`)
*   **Vector Query**: ChromaDB (Local) + HuggingFace Embeddings (`all-MiniLM-L6-v2`)
*   **Search Provider**: Google Jobs via SerpApi
*   **Runtime**: Python 3.12+ via `uv` package manager

## 7. Flow Diagram
_Refer to the Code Walkthrough Document for the detailed visual representation of the workflow._
