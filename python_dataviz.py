
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama3-70b-8192",temperature=0.2)

SYSTEM_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a python data visualizer which uses python repl tool to generate plotly visualization with title and labelled axis.
Generate the plot : {kind} for the features : {features} only.

NOTE:
1. Make the plot only with pandas,numpy,plotly.express and plotly.graph_objects libraries.
2. The generated plot should be clear and colorful with title and labeled axes.
3. Execute only 1 plot and show that plot in streamlit using plotly_chart function of streamlit.
4. Don't generate multiple charts.

Also, after execution of plot, interpret the graph with reference to the dataset and show it using streamlit write function.<|eot_id|>
"""

prompt_template = ChatPromptTemplate.from_template(template=SYSTEM_PROMPT)



def run_dataviz_agent(kind,features,dataframe,prompt=SYSTEM_PROMPT):
    python_repl_tool = PythonREPLTool(name="Python REPL",
    description="""A python shell which execute code for plotly data visualization using libraries like 
    pandas, numpy, plotly.express and plotly.graph_objects only. Also, you can execute streamlit code in shell.""".strip())

    tool_agent = create_pandas_dataframe_agent(
    llm=model.bind_tools(tools=[python_repl_tool]),
    df=dataframe,
    extra_tools=[python_repl_tool],
    agent_type="tool-calling",
    verbose=True)

    input_prompt = prompt.format(kind=kind,
                                 features=features)
    
    tool_agent.invoke(input_prompt)
    return