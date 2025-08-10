import argparse
import os
import sys
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import TypedDict, List, Optional, Annotated
import pandas as pd

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# PDF processing
try:
    import pdfplumber
except ImportError:
    print("Please install pdfplumber: pip install pdfplumber")
    sys.exit(1)


class AgentState(TypedDict):
    """State management for the agent workflow."""
    target_bank: str
    pdf_path: str
    csv_path: str
    parser_output_path: str
    pdf_content: str
    csv_schema: dict
    csv_sample_data: str
    generated_code: str
    test_results: dict
    error_message: str
    attempt_count: int
    messages: Annotated[List, add_messages]


class BankParserAgent:
    """Main agent class implementing the LangGraph workflow."""
    
    def __init__(self):
        self.llm = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            groq_api_key=os.getenv("GROQ_API_KEY", groq_key),
        )
        self.max_attempts = 3
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with conditional routing."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("plan", self.plan_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("test", self.test_node)
        workflow.add_node("reflect", self.reflect_node)
        
        # Add edges with conditional routing
        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "generate")
        workflow.add_edge("generate", "test")
        workflow.add_conditional_edges(
            "test",
            self.should_continue,
            {
                "continue": "reflect",
                "end": END
            }
        )
        workflow.add_edge("reflect", "generate")
        
        return workflow.compile()
    
    def plan_node(self, state: AgentState) -> AgentState:
        """Analyze requirements and create implementation strategy."""
        print(f"Planning parser for {state['target_bank']} bank...")
        
        # Extract PDF content
        pdf_content = self._extract_pdf_content(state['pdf_path'])
        
        # Analyze CSV structure
        csv_df = pd.read_csv(state['csv_path'])
        csv_schema = {
            'columns': csv_df.columns.tolist(),
            'dtypes': csv_df.dtypes.to_dict(),
            'shape': csv_df.shape,
            'sample_rows': csv_df.head(3).to_dict('records')
        }
        
        state['pdf_content'] = pdf_content
        state['csv_schema'] = csv_schema
        state['csv_sample_data'] = csv_df.to_string()
        state['attempt_count'] = 0
        
        plan_message = f"""
        Analysis complete for {state['target_bank']} bank:
        - PDF content extracted: {len(pdf_content)} characters
        - CSV schema: {len(csv_schema['columns'])} columns
        - Expected output format: {csv_schema['columns']}
        """
        
        state['messages'].append(SystemMessage(content=plan_message))
        print(f"Plan created. CSV columns: {csv_schema['columns']}")
        
        return state
    
    def generate_node(self, state: AgentState) -> AgentState:
        """Generate Python parser code using ChatGroq."""
        print(f"Generating parser code (attempt {state['attempt_count'] + 1})...")
        
        # Prepare prompt with context
        system_prompt = f"""You are an expert Python developer creating a bank statement PDF parser.

        Target Bank: {state['target_bank']}
        
        Requirements:
        1. Create a function parse(pdf_path) -> pd.DataFrame
        2. Extract data from PDF and return DataFrame matching the expected schema
        3. Use pdfplumber for PDF processing
        4. Handle errors gracefully
        5. Return DataFrame with exact column names as specified
        
        Expected CSV Schema:
        Columns: {state['csv_schema']['columns']}
        Sample data structure:
        {state['csv_sample_data'][:500]}...
        
        PDF Content Sample:
        {state['pdf_content'][:1000]}...
        
        Previous attempt feedback (if any):
        {state.get('error_message', 'None')}
        
        Generate ONLY the complete Python code for the parser file. Include all necessary imports.
        The code should be production-ready and handle edge cases.
        """
        
        user_prompt = f"Generate a complete {state['target_bank']}_parser.py file with parse(pdf_path) function."
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            generated_code = response.content
            
            # Clean up the code (remove markdown formatting if present)
            if "```python" in generated_code:
                generated_code = generated_code.split("```python").split("```")
            elif "```" in generated_code:
                generated_code = generated_code.split("``````")[0]
                
            
            state['generated_code'] = generated_code.strip()
            state['messages'].append(AIMessage(content=f"Generated parser code ({len(generated_code)} chars)"))
            
            print("Parser code generated successfully.")
            
        except Exception as e:
            error_msg = f"Code generation failed: {str(e)}"
            state['error_message'] = error_msg
            state['messages'].append(AIMessage(content=error_msg))
            print(f"Error: {error_msg}")
        
        return state
    
    def test_node(self, state: AgentState) -> AgentState:
        """Test the generated parser against expected output."""
        print("Testing generated parser...")
        
        state['attempt_count'] += 1
        
        try:
            # Write generated code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(state['generated_code'])
                temp_parser_path = temp_file.name
            
            # Test the parser in isolated environment
            test_script = f"""
import sys
sys.path.insert(0, '{os.path.dirname(temp_parser_path)}')
import pandas as pd
from pathlib import Path

# Import the generated parser
spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
    "temp_parser", "{temp_parser_path}"
)
temp_parser = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
spec.loader.exec_module(temp_parser)

# Test the parse function
try:
    result_df = temp_parser.parse("{state['pdf_path']}")
    
    # Load expected CSV
    expected_df = pd.read_csv("{state['csv_path']}")
    
    # Compare results
    print("PARSER_OUTPUT_SHAPE:", result_df.shape)
    print("EXPECTED_OUTPUT_SHAPE:", expected_df.shape)
    print("PARSER_COLUMNS:", list(result_df.columns))
    print("EXPECTED_COLUMNS:", list(expected_df.columns))
    
    # Check if DataFrames are equal
    if result_df.equals(expected_df):
        print("TEST_RESULT: PASS")
    else:
        print("TEST_RESULT: FAIL")
        print("DIFFERENCE_SUMMARY:")
        print("Columns match:", set(result_df.columns) == set(expected_df.columns))
        print("Shape match:", result_df.shape == expected_df.shape)
        
except Exception as e:
    print("PARSER_ERROR:", str(e))
    print("TEST_RESULT: ERROR")
"""
            
            # Run test in subprocess
            result = subprocess.run([sys.executable, '-c', test_script], 
                                  capture_output=True, text=True, timeout=30)
            
            # Parse test results
            output_lines = result.stdout.split('\n')
            test_passed = any('TEST_RESULT: PASS' in line for line in output_lines)
            
            if test_passed:
                # Save successful parser
                os.makedirs(os.path.dirname(state['parser_output_path']), exist_ok=True)
                with open(state['parser_output_path'], 'w') as f:
                    f.write(state['generated_code'])
                
                state['test_results'] = {
                    'passed': True,
                    'output': result.stdout,
                    'error': None
                }
                
                print("Test PASSED! Parser saved successfully.")
                
            else:
                error_info = result.stdout + result.stderr
                state['test_results'] = {
                    'passed': False,
                    'output': result.stdout,
                    'error': error_info
                }
                
                print(f"Test FAILED. Output: {result.stdout[:200]}...")
            
            # Cleanup
            os.unlink(temp_parser_path)
            
        except subprocess.TimeoutExpired:
            state['test_results'] = {
                'passed': False,
                'output': '',
                'error': 'Parser execution timed out'
            }
            print("Test FAILED: Timeout")
            
        except Exception as e:
            state['test_results'] = {
                'passed': False,
                'output': '',
                'error': f"Test execution error: {str(e)}\n{traceback.format_exc()}"
            }
            print(f"Test FAILED: {str(e)}")
        
        return state
    
    def reflect_node(self, state: AgentState) -> AgentState:
        """Analyze test failures and provide improvement feedback."""
        print("Reflecting on test failure and generating improvements...")
        
        reflection_prompt = f"""
        The generated parser for {state['target_bank']} failed testing. Analyze the issue and provide specific feedback.
        
        Generated Code:
        {state['generated_code'][:1500]}...
        
        Test Results:
        {state['test_results']['error']}
        {state['test_results']['output']}
        
        PDF Content Sample:
        {state['pdf_content'][:800]}...
        
        Expected CSV Schema:
        {state['csv_schema']}
        
        Provide specific feedback on what went wrong and how to fix it. Focus on:
        1. PDF parsing issues
        2. Data extraction problems
        3. DataFrame structure mismatches
        4. Column naming or data type issues
        """
        
        try:
            response = self.llm.invoke([
                SystemMessage(content="You are an expert code reviewer analyzing parser failures."),
                HumanMessage(content=reflection_prompt)
            ])
            
            state['error_message'] = response.content
            state['messages'].append(AIMessage(content=f"Reflection complete: {response.content[:200]}..."))
            
            print(f"Reflection complete. Key issues identified.")
            
        except Exception as e:
            state['error_message'] = f"Reflection failed: {str(e)}"
            print(f"Reflection error: {str(e)}")
        
        return state
    
    def should_continue(self, state: AgentState) -> str:
        """Determine whether to continue with corrections or end."""
        if state['test_results']['passed']:
            return "end"
        elif state['attempt_count'] >= self.max_attempts:
            print(f"Maximum attempts ({self.max_attempts}) reached. Ending workflow.")
            return "end"
        else:
            print(f"Test failed. Attempting correction ({state['attempt_count']}/{self.max_attempts})...")
            return "continue"
    
    def _extract_pdf_content(self, pdf_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                content = []
                for page in pdf.pages[:3]:  # Extract first 3 pages
                    text = page.extract_text()
                    if text:
                        content.append(text)
                return '\n'.join(content)
        except Exception as e:
            return f"PDF extraction failed: {str(e)}"
    
    def run(self, target_bank: str) -> bool:
        """Execute the complete workflow for the target bank."""
        print(f"Starting Agent-as-Coder workflow for {target_bank} bank...")
        
        # Setup paths
        data_dir = Path(f"data/{target_bank}")
        pdf_path = data_dir / f"{target_bank}_sample.pdf"
        csv_path = data_dir / f"{target_bank}_sample.csv"
        parser_output_path = Path(f"custom_parsers/{target_bank}_parser.py")
        
        # Create directories if they don't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        parser_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if sample files exist
        if not pdf_path.exists() or not csv_path.exists():
            print(f"Sample files not found. Please ensure these files exist:")
            print(f"- {pdf_path}")
            print(f"- {csv_path}")
            self._create_sample_files(data_dir, target_bank)
            return False
        
        # Initialize state
        initial_state = AgentState(
            target_bank=target_bank,
            pdf_path=str(pdf_path),
            csv_path=str(csv_path),
            parser_output_path=str(parser_output_path),
            pdf_content="",
            csv_schema={},
            csv_sample_data="",
            generated_code="",
            test_results={},
            error_message="",
            attempt_count=0,
            messages=[]
        )
        
        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            if final_state['test_results'].get('passed', False):
                print(f"SUCCESS: Parser generated and saved to {parser_output_path}")
                return True
            else:
                print(f"FAILED: Could not generate working parser after {self.max_attempts} attempts")
                print(f"Last error: {final_state.get('error_message', 'Unknown error')}")
                return False
                
        except Exception as e:
            print(f"Workflow execution failed: {str(e)}")
            traceback.print_exc()
            return False
    
    def _create_sample_files(self, data_dir: Path, bank_name: str):
        """Create sample files for demonstration."""
        print(f"Creating sample files for {bank_name}...")
        
        # Create sample CSV
        sample_csv = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Description': ['Transfer', 'ATM Withdrawal', 'Online Purchase'],
            'Amount': [1000.0, -500.0, -250.0],
            'Balance': [10000.0, 9500.0, 9250.0]
        })
        
        csv_path = data_dir / f"{bank_name}_sample.csv"
        sample_csv.to_csv(csv_path, index=False)
        
        print(f"Sample CSV created at {csv_path}")
        print("Please add the corresponding PDF file and run again.")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Agent-as-Coder: Bank Statement Parser Generator")
    parser.add_argument("--target", required=True, help="Target bank name (e.g., icici, sbi)")
    
    args = parser.parse_args()
    
    # Initialize and run agent
    agent = BankParserAgent()
    success = agent.run(args.target)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
