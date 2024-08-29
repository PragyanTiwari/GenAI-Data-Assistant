
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pandas_df_agent import DataframeAgent
from python_dataviz import run_dataviz_agent
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

st.header("AI Assistant for Data Analyst",divider="rainbow")
st.write("An AI Assitant for Data Analaysis and Exploration of datasets.")

tab_home, tab_EDA, tab_Viz, tab_Chat = st.tabs(["Home", "Explore Dataset", "Visualize Relationships", "ChatBox"])


#------------------------------------------------------------------------------------------------------------------------------------

with tab_home:
    st.markdown("##### Hi User!")
    st.markdown("##### Meet an AI Assistant which help in you data analysis tasks.")
    st.markdown("###### You can explore data, Visualize relationships of features and Chat with the :blue[Gemini API]")
    
    st.write("##### To start your exploration, click the below button")
    
    if "clicked" not in st.session_state:
     st.session_state["clicked"] = False
    def updating_click():
        st.session_state["clicked"] = True
        
    st.button("Let's get started",on_click=updating_click)
    if st.session_state["clicked"] == True:
        input_csv_file = st.file_uploader("Attach the data in csv format",type="csv")
        if input_csv_file:
            df = pd.read_csv(input_csv_file,low_memory=False)       

            # session state
            with st.spinner("Initializing pandas agent"):
                pd_agent_obj = DataframeAgent(data=df)
                pd_response = pd_agent_obj.run_chain()
                columns = df.columns.tolist()
                st.success("Done!")
                st.session_state["collect_pd_response"] = True

# ----------------------------------------------------------------------------------------------------------------------------------------
if "collect_pd_response" in st.session_state:

    with tab_EDA:    
            st.markdown("#### Explore the structure of the data")
            st.dataframe(df)

            st.markdown("#### Feature Exploration")
            st.markdown(pd_response.feature_Explanation)
            st.markdown(f"Shape of the dataframe is : {pd_response.shape}")
            st.markdown(f"No. of duplicate rows: {pd_response.n_duplicates}")

            st.markdown("#### Correlation Matrix")
            numdf_corr = df.select_dtypes(exclude="object").corr()
            st.table(data=numdf_corr)
            st.markdown(pd_response.correlation)
                    
            st.markdown("#### Data types of the dataframe")
            st.markdown(pd_response.feature_data_types)

            st.markdown("#### Statistics Summary")
            st.table(data=df.describe())
            st.markdown(pd_response.summarize_statistics)            

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    analysis_dct = {
        "Univariate":("Bar plot","Histogram plot","Box plot","Violin plot","Stacked Histogram"),
        "Bivariate":("Scatter plot","Line plot","Box plot","Bubble plot"),
        "Multivariate":("Box plot","Violin plot","Scatter plot","Line plot")
        }

    with tab_Viz:

        analysis = st.selectbox("What kind of Analysis you want?",
                    ("Univariate","Bivariate","Multivariate"))
        variables,plot = st.columns(2)

        def var_plot_options(analysis):
            with variables:
                var = st.multiselect("Select the feature to plot",columns)
            with plot:
                plot_kind = st.selectbox("Select the kind to plot",analysis_dct[analysis])
            return var,plot_kind
        inp_var,inp_plot = var_plot_options(analysis=analysis)

        if st.button("click to generate plot"):
            run_dataviz_agent(kind=inp_plot,features=inp_var,dataframe=df)

    # -----------------------------------------------------------------------------------------------------------------------------------------------
    
    with tab_Chat:
        prompt = st.chat_input("enter your message")
        #initializing chatbot session state
        def initialize_session_state():
            if "messages" not in st.session_state:
                st.session_state["messages"] = []
                greeting_message = {"role": "assistant", "content": "Hey! I'm pandas AI Assistant"}
                st.session_state["messages"].append(greeting_message)
        
        def display_chat_history():
            for message in st.session_state["messages"]:
                with st.chat_message(name=message["role"]):
                    st.markdown(message["content"])
        
        def get_user_prompt(text: str):
            user_message = {"role": "user", "content": text}
            with st.chat_message(name=user_message['role']):
                st.markdown(text)
            return user_message
        
        def get_llm_response(input_prompt: str, df):
            prompt_message = ChatPromptTemplate.from_messages([
                SystemMessage(content="You're a chatbot. Your task is to have a conversation with the user and answer the user question."),
                HumanMessage(content=input_prompt)])

            agent = create_pandas_dataframe_agent(llm=llm, df=df, verbose=True)
            agent_response = agent.invoke(prompt_message)

            llm_message = {"role": "assistant", "content": agent_response['output']}
            return llm_message
        
        
        initialize_session_state()
        display_chat_history()

        if prompt:
            user_says = get_user_prompt(text=prompt)
            st.session_state['messages'].append(user_says)

            llm_says = get_llm_response(input_prompt=prompt, df=df)
            with st.chat_message(llm_says['role']):
                st.markdown(llm_says["content"])
            st.session_state['messages'].append(llm_says)