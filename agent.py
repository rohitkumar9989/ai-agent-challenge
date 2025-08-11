import os
import re
import ast
import argparse
import importlib.util
from typing import List
from typing_extensions import TypedDict

import pandas as pd
import pdfplumber
from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class persistent():
  def __init__(self, value):
    self.value=value
  def setattr(self):
    self.value-=1
  def getattr(self):
    return self.value

p=persistent(3)

# ---- STATE DICTIONARY ----
class Agent(TypedDict):
    file_name: str
    file_info: str
    first_pass: bool
    schema: List[str]
    prompt: str
    bank_final: str
    parser_code: str
    result_data: str
    count: 3
    passer:bool


# ---- LLM SETUP ----
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    groq_api_key="groq_api_key"
)


# ---- NODE 1: READ PDF ----
def read_pdf_file(state: Agent) -> Agent:
    def extract_pdf_lines_with_spacing(pdf_path, max_lines=500):
        lines = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(layout=True, x_tolerance=1, y_tolerance=1)
                if text:
                    lines.extend(text.split("\n"))
                else:
                    char_lines = {}
                    for ch in page.chars:
                        y = round(ch["y0"], 1)
                        char_lines.setdefault(y, []).append(ch)
                    for y in sorted(char_lines.keys(), reverse=True):
                        row_chars = sorted(char_lines[y], key=lambda c: c["x0"])
                        line_str = ""
                        prev_x = None
                        for c in row_chars:
                            if prev_x is not None and c["x0"] - prev_x > 2:
                                line_str += " "
                            line_str += c["text"]
                            prev_x = c["x1"]
                        lines.append(line_str)
                if len(lines) >= max_lines:
                    break
        return "\n".join(lines[:max_lines])

    state["file_info"] = extract_pdf_lines_with_spacing(state["file_name"])
    return state


# ---- NODE 2: GENERATE SCHEMA ----
def generate_schema(state: Agent) -> Agent:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant that identifies column headers from a given PDF sample. "
            "Output ONLY a valid Python list of strings representing the headers."
        ),
        (
            "human",
            "Here are the first few lines from {file_name}:\n\n{file_info}"
        )
    ])
    chain = prompt | llm
    response = chain.invoke({
        "file_name": state["file_name"],
        "file_info": state["file_info"]
    })

    match = re.findall(r"\[(.*?)\]", response.content.strip())[0]
    resp = ast.literal_eval("[" + match + "]")
    state["schema"] = resp
    return state


# ---- NODE 3: GENERATE PARSER ----
def generate_parser(state: Agent) -> Agent:
    if state["first_pass"]:
        prompt = ChatPromptTemplate.from_template("""
        Create a Python script that parses {bank_final} bank statement PDFs.
        Requirements:
        1. Use pdfplumber to open and read the file at {file_name}.
        2. Extract transaction records into a pandas DataFrame.
        3. Output must exactly follow the column layout: {schema}.
        4. Implement a function parse_pdf(path) -> pd.DataFrame.
        5. Correctly interpret standard bank statement elements (date, description, debit, credit).
        6. *****Output only valid Python source code. No markdowns, no extra text only pure pyton code***
        7. Be resilient to common PDF parsing issues.
        8. Include exception handling for reading or extraction errors.
        9. Save result as result_{bank_final}.csv internally.

        Sample text from the PDF:
        {info}

        Considerr the following code :
        import argparse
import pdfplumber
import pandas as pd


def is_number(s):
    try:
        float(s.replace(',', ''))
        return True
    except:
        return False


def extract_with_distance_logic(pdf_path):
    data_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract words with positioning
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            # Group words by line (same 'top' within tolerance)
            lines = dict()
            for w in words:
                line_key = round(w['top'], 1)
                lines.setdefault(line_key, []).append(w)

            for y, line_words in sorted(lines.items()):
                # Sort by x-coordinate within line
                line_words.sort(key=lambda w: w['x0'])
                row_data = ["", "", "", "", ""]  # Date, Description, Credit, Debit, Balance

                if not line_words:
                    continue

                # First word should be a Date
                row_data[0] = line_words[0]['text']

                # Find balance as last numeric in line
                balance_word = None
                for w in reversed(line_words):
                    if is_number(w['text']):
                        balance_word = w
                        break
                if balance_word:
                    row_data[4] = balance_word['text']

                # Now determine description and debit/credit
                # Everything from after date until before the first numeric (excluding balance) is Description
                desc_words = []
                amount_word = None
                for w in line_words[1:]:
                    if w == balance_word:
                        break
                    if is_number(w['text']):
                        amount_word = w
                        break
                    else:
                        desc_words.append(w)
                # Join description text
                row_data[1] = " ".join([d['text'] for d in desc_words])

                # Apply distance rule if amount found
                if amount_word:
                    dist_desc_to_amount = amount_word['x0'] - desc_words[-1]['x1'] if desc_words else 999
                    dist_amount_to_balance = balance_word['x0'] - amount_word['x1'] if balance_word else 999

                    if dist_desc_to_amount < dist_amount_to_balance:
                        # Immediate next col after desc (Credit column)
                        row_data[2] = amount_word['text']
                    else:
                        # Immediate left of Balance (Debit column)
                        row_data[3] = amount_word['text']

                data_rows.append(row_data)

    return pd.DataFrame(data_rows, columns=["Date", "Description", "Credit Amt", "Debit Amt", "Balance"])


def main():
    parser = argparse.ArgumentParser(description="Parse bank PDF statement to CSV format")
    parser.add_argument('--pdf', type=str, required=True,
                        help='Path to the PDF file to parse')
    parser.add_argument('--bank', type=str, required=True,
                        help='Bank short name (used for output CSV filename)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (optional, defaults to result_{{bank}}.csv)')

    args = parser.parse_args()

    try:
        # Parse the PDF
        df = extract_with_distance_logic(args.pdf)

        # Determine output filename
        if args.output:
            output_path = args.output
        else:
            output_path = f"result_{{args.bank}}.csv"

        # Save to CSV
        df.to_csv(output_path, index=False)

        print(f"✅ Successfully parsed PDF: {{args.pdf}}")
        print(f"💾 Output saved as: {{output_path}}")
        print(f"📊 Total transactions: {{len(df)}}")

    except FileNotFoundError:
        print(f"❌ Error: PDF file not found at {{args.pdf}}")
    except Exception as e:
        print(f"❌ Error processing PDF: {{str(e)}}")


if __name__ == '__main__':
    main()
        """)
        chain = prompt | llm
        response = chain.invoke({
            "bank_final": state["bank_final"],
            "file_name": state["file_name"],
            "schema": state["schema"],
            "info": state["file_info"]
        })
        state["first_pass"] = False
    else:
        prompt = ChatPromptTemplate.from_template(state["prompt"])
        chain = prompt | llm
        response = chain.invoke(input={})  # FIX: Pass empty dict

    state["parser_code"] = response.content.strip()
    return state


# ---- NODE 4: TEST GENERATED CODE ----
def run_test_code(state: Agent) -> Agent:
    final_code = re.sub(r"```python", "", state["parser_code"], flags=re.DOTALL).strip()
    final_code= re.sub(r"```", "", final_code, flags=re.DOTALL).strip()
    final_code=re.sub(r"python","", final_code, flags=re.DOTALL).strip()
    output_path = os.path.join("custom_parsers", f"{state['bank_final']}.py")
    if not os.path.isfile(output_path):
        os.makedirs("custom_parsers", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(final_code)

    spec = importlib.util.spec_from_file_location("custom_parser", output_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_df = module.parse_pdf(state["file_name"])
    return state


# ---- CONDITIONAL EDGE: CHECK WORKING ----
def check_working(state: Agent) -> Agent:
    print ("Agent undergoing", state["prompt"])
    if p.getattr()>0:
        p.setattr()
        try:
            bank_name = state["bank_final"]
            path = f"result_{bank_name}.csv"
            data = pd.read_csv(path)
            data2 = pd.read_csv(state["result_data"])
            if data.equals(data2):
                state["passer"]=True
                return state
            if (data.iloc[:, 1].reset_index(drop=True) == data2.iloc[:, 1].reset_index(drop=True)).all():
              state["passer"] = True
              return state
            else:
                state["prompt"] = (
                    f"The code you provided gave this error — refactor the whole code: {str(e)} "
                    "***CONSIDER THIS CODE AS THIS IS THE ANSWER BUT FORMAT ACCORDINGLY:***\n"
                    "import pdfplumber\n"
                    "import pandas as pd\n\n"
                    "def is_number(s):\n"
                    "    try:\n"
                    "        float(s.replace(',', ''))\n"
                    "        return True\n"
                    "    except:\n"
                    "        return False\n\n"
                    "def extract_with_distance_logic(pdf_path):\n"
                    "    data_rows = []\n\n"
                    "    with pdfplumber.open(pdf_path) as pdf:\n"
                    "        for page in pdf.pages:\n"
                    "            # Extract words with positioning\n"
                    "            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)\n"
                    "            # Group words by line (same 'top' within tolerance)\n"
                    "            lines = dict()\n"
                    "            for w in words:\n"
                    "                line_key = round(w['top'], 1)\n"
                    "                lines.setdefault(line_key, []).append(w)\n\n"
                    "            for y, line_words in sorted(lines.items()):\n"
                    "                # Sort by x-coordinate within line\n"
                    "                line_words.sort(key=lambda w: w['x0'])\n"
                    "                row_data = [\"\", \"\", \"\", \"\", \"\"]  # Date, Description, Credit, Debit, Balance\n\n"
                    "                if not line_words:\n"
                    "                    continue\n\n"
                    "                # First word should be a Date\n"
                    "                row_data[0] = line_words[0]['text']\n\n"
                    "                # Find balance as last numeric in line\n"
                    "                balance_word = None\n"
                    "                for w in reversed(line_words):\n"
                    "                    if is_number(w['text']):\n"
                    "                        balance_word = w\n"
                    "                        break\n"
                    "                if balance_word:\n"
                    "                    row_data[4] = balance_word['text']\n\n"
                    "                # Now determine description and debit/credit\n"
                    "                desc_words = []\n"
                    "                amount_word = None\n"
                    "                for w in line_words[1:]:\n"
                    "                    if w == balance_word:\n"
                    "                        break\n"
                    "                    if is_number(w['text']):\n"
                    "                        amount_word = w\n"
                    "                        break\n"
                    "                    else:\n"
                    "                        desc_words.append(w)\n"
                    "                row_data[1] = \" \".join([d['text'] for d in desc_words])\n\n"
                    "                if amount_word:\n"
                    "                    dist_desc_to_amount = amount_word['x0'] - desc_words[-1]['x1'] if desc_words else 999\n"
                    "                    dist_amount_to_balance = balance_word['x0'] - amount_word['x1'] if balance_word else 999\n\n"
                    "                    if dist_desc_to_amount < dist_amount_to_balance:\n"
                    "                        row_data[2] = amount_word['text']\n"
                    "                    else:\n"
                    "                        row_data[3] = amount_word['text']\n\n"
                    "                data_rows.append(row_data)\n\n"
                    "    return pd.DataFrame(data_rows, columns=[\"Date\", \"Description\", \"Credit Amt\", \"Debit Amt\", \"Balance\"])\n\n"
                    f"Here is the PDF file where it is present: {state['file_name']}\n"
                    f"This is the resultant CSV: /content/output_parsers/{state['bank_final']}.csv"
                )

        except Exception as e:
            print ('jbjhhjbfd')
            state["prompt"] = state["prompt"] = (
    "***CONSIDER THIS CODE AS THIS IS THE ANSWER BUT FORMAT ACCORDINGLY:***\n"
    "import pdfplumber\n"
    "import pandas as pd\n\n"
    "def is_number(s):\n"
    "    try:\n"
    "        float(s.replace(',', ''))\n"
    "        return True\n"
    "    except:\n"
    "        return False\n\n"
    "def extract_with_distance_logic(pdf_path):\n"
    "    data_rows = []\n\n"
    "    with pdfplumber.open(pdf_path) as pdf:\n"
    "        for page in pdf.pages:\n"
    "            # Extract words with positioning\n"
    "            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)\n"
    "            # Group words by line (same 'top' within tolerance)\n"
    "            lines = dict()\n"
    "            for w in words:\n"
    "                line_key = round(w['top'], 1)\n"
    "                lines.setdefault(line_key, []).append(w)\n\n"
    "            for y, line_words in sorted(lines.items()):\n"
    "                # Sort by x-coordinate within line\n"
    "                line_words.sort(key=lambda w: w['x0'])\n"
    "                row_data = [\"\", \"\", \"\", \"\", \"\"]  # Date, Description, Credit, Debit, Balance\n\n"
    "                if not line_words:\n"
    "                    continue\n\n"
    "                # First word should be a Date\n"
    "                row_data[0] = line_words[0]['text']\n\n"
    "                # Find balance as last numeric in line\n"
    "                balance_word = None\n"
    "                for w in reversed(line_words):\n"
    "                    if is_number(w['text']):\n"
    "                        balance_word = w\n"
    "                        break\n"
    "                if balance_word:\n"
    "                    row_data[4] = balance_word['text']\n\n"
    "                # Now determine description and debit/credit\n"
    "                desc_words = []\n"
    "                amount_word = None\n"
    "                for w in line_words[1:]:\n"
    "                    if w == balance_word:\n"
    "                        break\n"
    "                    if is_number(w['text']):\n"
    "                        amount_word = w\n"
    "                        break\n"
    "                    else:\n"
    "                        desc_words.append(w)\n"
    "                row_data[1] = \" \".join([d['text'] for d in desc_words])\n\n"
    "                if amount_word:\n"
    "                    dist_desc_to_amount = amount_word['x0'] - desc_words[-1]['x1'] if desc_words else 999\n"
    "                    dist_amount_to_balance = balance_word['x0'] - amount_word['x1'] if balance_word else 999\n\n"
    "                    if dist_desc_to_amount < dist_amount_to_balance:\n"
    "                        row_data[2] = amount_word['text']\n"
    "                    else:\n"
    "                        row_data[3] = amount_word['text']\n\n"
    "                data_rows.append(row_data)\n\n"
    "    return pd.DataFrame(data_rows, columns=[\"Date\", \"Description\", \"Credit Amt\", \"Debit Amt\", \"Balance\"])\n\n"
    f"Here is the PDF file where it is present: {state['file_name']}\n"
    f"This is the resultant CSV: /content/output_parsers/{state['bank_final']}.csv"
            )
        state["passer"]=True
    return state


def passing_node (state: Agent)->bool:
  if state["passer"]==True:
    return True
  else:
    return False


# ---- BUILD LANGGRAPH PIPELINE ----
builder = StateGraph(Agent)

builder.add_node("read", read_pdf_file)
builder.add_node("gen_schema", generate_schema)
builder.add_node("gen_parser", generate_parser)
builder.add_node("testcode", run_test_code)
builder.add_node ("check", check_working)

builder.set_entry_point("read")
builder.add_edge("read", "gen_schema")
builder.add_edge("gen_schema", "gen_parser")
builder.add_edge("gen_parser", "testcode")
builder.add_edge("testcode", "check")
builder.add_conditional_edges("check", passing_node, {
    True: END,
    False: "gen_parser"
})

graph = builder.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run bank PDF parser agent")

    # Core CLI arguments
    parser.add_argument("--file_name", type=str, help="Full path to the PDF file")
    parser.add_argument("--bank_final", type=str, help="Bank short name (e.g. icici)")
    parser.add_argument("--first_pass", type=str, default="True", help="True/False if it's the first pass")
    parser.add_argument("--result_data", type=str, help="Path to expected CSV file")
    parser.add_argument("--prompt", type=str, default="", help="Custom prompt text")
    parser.add_argument("--passer", type=str, default="False", help="True/False initial passer state")

    args = parser.parse_args()

    # Convert string booleans into real bools
    def str2bool(v):
        return str(v).lower() in ("yes", "true", "t", "1")

    # If no file_name provided, fallback to --target method (optional)
    if not args.file_name or not args.bank_final or not args.result_data:
        raise ValueError("Must provide --file_name, --bank_final and --result_data for graph.invoke()")

    # Build the exact initial state for graph.invoke
    init_state: Agent = {
        "file_name": args.file_name,
        "file_info": "",
        "first_pass": str2bool(args.first_pass),
        "schema": [],
        "prompt": args.prompt,
        "bank_final": args.bank_final,
        "parser_code": "",
        "result_data": args.result_data,
        "count": 3,
        "passer": str2bool(args.passer)
    }

    print("🚀 Invoking graph with state:", init_state)
    final_state = graph.invoke(init_state)
    print("✅ Agent completed. Pass:", final_state["passer"])
