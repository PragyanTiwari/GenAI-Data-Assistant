import pandas as  pd
from langchain_groq import ChatGroq
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.prompts import ChatPromptTemplate
from langchain_experimental.tools import PythonREPLTool
from dotenv import load_dotenv

load_dotenv()
df = pd.read_csv(r"E:\Machine learning\Sources\telecom data.csv")
model = ChatGroq(model="llama3-70b-8192")

SYSTEM_PROMPT = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a python data visualizer which uses python repl tool to generate plotly visualization with title and labelled axis.
Generate the plot : {kind} for the features : {features} only.

NOTE:
1. Make the plot only with pandas,numpy,plotly.express,streamlit libraries.
2. The generated plot should be clear and colorful with title and labeled axes.
3. Execute only 1 plot in streamlit using st.plotly_chart function
4. Don't generate multiple charts.

Also, after execution of plot, interpret the graph with reference to the dataset and show it using streamlit write function.<|eot_id|>
"""

prompt_template = ChatPromptTemplate.from_template(template=SYSTEM_PROMPT)


def run_dataviz_agent(kind,features,dataframe,prompt=SYSTEM_PROMPT):
    python_repl_tool = PythonREPLTool(name="Python REPL",
    description="""A python shell which execute code for data visualization using the following libraries i.e. 
    pandas, numpy, streamlit and plotly.express only. Execute streamlit code in shell to display the graph using plotly_chart function.""".strip())

    tool_agent = create_pandas_dataframe_agent(
    llm=model.bind_tools(tools=[python_repl_tool]),
    df=dataframe,
    extra_tools=[python_repl_tool],
    agent_type="tool-calling",
    allow_dangerous_code=True,
    verbose=True)

    input_prompt = prompt.format(kind=kind,
                                 features=features)
    
    tool_agent.invoke(input_prompt)
    return

code = run_dataviz_agent(kind="scatter plot",features="tenure,income",dataframe=df)
print(code)