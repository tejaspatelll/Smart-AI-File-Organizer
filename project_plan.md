# Smart File Organizer Project Plan

## Project Overview

A desktop application powered by Google Gemini to intelligently organize files based on their metadata and content, creating relevant folders and moving files accordingly. The goal is to automate and streamline file management for users dealing with large numbers of unorganized files.

## Key Features

*   **File Scanning:** Recursively scan specified directories for files, intelligently skipping irrelevant directories like `node_modules`.
*   **Metadata Extraction:** Extract various metadata (e.g., creation date, file type, size, author) from files.
*   **Content Analysis (Gemini Integration):** Utilize Google Gemini to analyze file content (for supported file types like documents, text files) to understand context, identify themes, and categorize files intelligently.
*   **Intelligent Folder Creation:** Automatically create new, descriptive folders based on organizing rules derived from metadata and content analysis (e.g., "Invoices_2023", "Photos_Summer_Vacation_2024", "Project_X_Documents").
*   **File Movement & Organization:** Move files to their respective, newly created or existing organized folders while preserving existing folder structures where appropriate.
*   **User Interface:** A user-friendly graphical interface with a six-step guided workflow (Welcome, Directory Selection, Method Selection, Processing, Results Preview, and Success) to:
    *   Select directories to organize.
    *   Choose from four distinct organization methods: Quick & Smart (zero-configuration AI), Custom Prompts (natural language instructions), Professional Templates (pre-built structures), and Advanced Configuration (fine-grained control).
    *   View proposed organization changes in a highly enhanced dry-run mode with Table, Cards, and Grouped views, real-time search, filter controls, Quick Actions dropdown, and a clear "FROM → TO" visual flow.
    *   Approve, modify, or selectively apply proposed changes before execution.
    *   Monitor organizing progress with animated progress bars and view a detailed summary.
*   **Customizable Rules:** Integrated within the Advanced Configuration method, allowing users to define and prioritize custom organizing rules or modify AI-suggested ones.
*   **Duplicate File Detection (Stretch Goal):** Identify and optionally manage (e.g., delete, move to quarantine) duplicate files.
*   **AI Confidence Scoring:** Display AI confidence levels for proposed file movements, allowing users to prioritize and review files needing higher attention.

## Tech Stack

*   **Backend/Core Logic:** Python
    *   **File System Operations:** `os`, `shutil`
    *   **Metadata Extraction:** Libraries like `Pillow` (for images), `python-docx` (for Word documents), `PyPDF2` (for PDFs), `exiftool` (potentially via subprocess for comprehensive metadata).
    *   **AI Integration:** `google-generativeai` (for Gemini API interaction), `CustomPromptClassifier` for natural language understanding.
    *   **Other Utilities:** `tqdm` (for progress bars), `json` (for config/rules).
*   **Frontend (Desktop Application):** Electron (Node.js with web technologies - HTML, CSS, JavaScript)
    *   **Rationale:** Provides a robust, cross-platform desktop application experience with familiar web development tools, allowing for rich and interactive UI. Features Apple-inspired design, glassmorphism, consistent design system with CSS custom properties, and smooth animations.
*   **AI Model:** Google Gemini API (for advanced content understanding and categorization).
*   **Database (Optional):** SQLite (for persistent storage of user preferences, custom rules, and operation history).

## Project Plan (Phased Approach)

### Phase 1: Core Functionality (Completed & Enhanced)

1.  **Environment Setup:** Initialized Python project structure, virtual environment, and basic Electron setup. Configured `google-generativeai` client.
2.  **Basic File Operations:** Implemented recursive directory scanning (now with intelligent filtering) and safe file movement.
3.  **Basic Metadata-based Organizing:** Extracted fundamental metadata and implemented initial organizing logic.
4.  **Initial Gemini Integration:** Developed functions to send file content to Gemini for basic categorization.
5.  **Basic User Interface:** Transformed into a comprehensive 6-step guided workflow with Apple-inspired design and smooth transitions.

### Phase 2: Intelligent Organizing & UI Enhancements (In Progress/Substantially Completed)

1.  **Advanced Metadata & Content Extraction:** Enhanced metadata extraction and robust text extraction from various document types.
2.  **Refined Gemini Interaction:** Crafted sophisticated prompts for Gemini, implemented `CustomPromptClassifier`, and added professional organization templates.
3.  **Interactive User Review & Approval:** Implemented a significantly enhanced "Review Your Organization Plan" screen with multiple views (Table, Cards, Grouped), advanced controls (search, filter, quick actions), and clear visual flow indicators.
4.  **Robust Error Handling & Logging:** Basic error handling implemented; further enhancements ongoing.

### Phase 3: Advanced Features & Optimizations (Future)

1.  **Customizable Rule Engine:** Fully integrating a dedicated UI for managing complex custom rules. (Currently part of Advanced Configuration method)
2.  **Duplicate File Management:** Implementing robust duplicate file detection and management.
3.  **Performance & Scalability:** Further optimizing for very large directories and exploring multi-threading/asynchronous processing.
4.  **Settings and Configuration:** Creating a comprehensive settings panel for advanced preferences.
5.  **Application Packaging & Distribution:** Packaging the Electron application for various operating systems.