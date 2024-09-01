import json
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv

load_dotenv()
model = ChatGroq(model="llama3-70b-8192",temperature=0.2)

# retrieving the prompt from json file
with open("prompt.json","r") as system_file:
    data = json.load(system_file)
SYSTEM_PROMPT = data['SYSTEM_PROMPT']

prompt_template = ChatPromptTemplate.from_template(template=SYSTEM_PROMPT)

# creating the tool
python_repl_tool = PythonREPLTool(name="Python REPL",verbose=True,
    description="""A python shell which execute code for data visualization using the following libraries i.e. 
    pandas, numpy, streamlit and plotly.express only. Execute streamlit code in shell to display the graph using plotly_chart function.""".strip(),
    handle_tool_error=True,handle_validation_error=True)

model_with_tools = model.bind_tools(tools=[python_repl_tool])


# building the dataviz agent
def run_dataviz_agent(dataframe,
                      kind:str,features:str,
                      llm=model_with_tools,
                      prompt=SYSTEM_PROMPT):
    """"
    An agent to collect & execute python code.
    Args:
        kind:str = type of plot.
        features:str = list of features to use.
    
    Returns:
        execute the generated python code in the same shell.
    """
    df = dataframe
    input_prompt = prompt.format(kind=kind,features=features)

    # collecting model response
    model_response = llm.invoke(input=input_prompt)

    # collecting the tool response
    for tool_call in model_response.tool_calls:
        if tool_call['name'] == "Python REPL":
            tool_args = tool_call['args']['query']
            # executing tool code
            return exec(tool_args)
