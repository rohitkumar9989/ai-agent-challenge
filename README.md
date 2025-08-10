# Agent-as-Coder Challenge Solution

A LangGraph-based coding agent that generates custom parsers for bank statement PDFs with self-correction capabilities.

## Overview

This agent implements the "Agent-as-Coder" challenge requirements:
- Uses LangGraph for workflow orchestration
- ChatGroq with meta-llama/llama-4-scout-17b-16e-instruct model
- Plan → Generate Code → Run Tests → Self-Fix loop (≤3 attempts)
- Generates custom parsers that match CSV output exactly using DataFrame.equals()

## Installation

`pip install -r requirements.txt`


## Setup

Set your Groq API key (optional - fallback key included):
`export GROQ_API_KEY="your-groq-api-key"`


Ensure you have the required directory structure (auto-created):
`
data/
{target}/
{target}_sample.pdf
{target}_expected.csv
custom_parsers/
{target}_parser.py # Generated output
`

## Usage

Generate ICICI bank parser
`python agent.py --target icici

Generate SBI bank parser
`python agent.py --target sbi`

Generate any bank parser
`python agent.py --target {bank_name}`



## How It Works

The agent follows a structured workflow:

1. **Plan**: Analyzes requirements and creates implementation strategy
2. **Generate**: Creates Python code for the PDF parser
3. **Test**: Runs automated tests using DataFrame.equals() comparison
4. **Reflect**: If tests fail, analyzes errors and provides feedback
5. **Iterate**: Repeats generate→test→reflect up to 3 times until success

## Output

The agent generates a parser at `custom_parsers/{target}_parser.py` with:
- `parse(pdf_path) -> pd.DataFrame` function
- Proper imports and error handling
- Output matching expected CSV schema exactly
- Robust PDF text extraction using pdfplumber

## Challenge Requirements

- **T1**: LangGraph agent with plan→generate→test→self-fix loop
- **T2**: CLI interface with `python agent.py --target icici`
- **T3**: Parser contract `parse(pdf_path) -> pd.DataFrame`
- **T4**: Automated testing with `DataFrame.equals()` for exact matching
- **T5**: Self-correction capabilities with up to 3 attempts
- **T6**: Robust PDF parsing using appropriate libraries

## Features

-  Automated code generation for bank statement parsers
- Self-correcting workflow with error analysis
-  Exact CSV output matching verification
-  Support for multiple bank formats
-  Robust error handling and logging
-  Extensible architecture for new banks

## Architecture

The solution uses LangGraph to orchestrate a multi-step workflow:
- **Planning Node**: Strategy formulation
- **Code Generation Node**: Python parser creation
- **Testing Node**: Automated validation
- **Reflection Node**: Error analysis and correction
- **Conditional Routing**: Success/failure path management

## Technical Implementation

### Core Components

1. **LangGraph Workflow**: State machine managing the agent's execution flow
2. **ChatGroq Integration**: LLM-powered code generation and analysis
3. **PDF Processing**: pdfplumber-based text extraction
4. **Testing Framework**: Automated validation using pandas DataFrame comparison
5. **Self-Correction Loop**: Iterative improvement with error feedback

### State Management

The agent maintains state across workflow nodes:
- Target bank identifier
- Generated code iterations
- Test results and error messages
- Attempt counter for self-correction limits

## Dependencies

`langgraph
langchain
langchain-groq
pandas
pdfplumber
argparse
os
sys`



## Error Handling

- Graceful fallback for missing API keys
- Comprehensive error logging
- Automatic directory creation
- PDF parsing error recovery
- Test failure analysis and reporting
