
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
load_dotenv()

class DataframeInfoRetrive(BaseModel):
    """output response class"""
    feature_Explanation:str = Field(...,description="info about the explanation of each column of the feature with their unique vals with markdown syntax.")
    shape:tuple = Field(...,description="shape of the dataframe")
    n_duplicates:int = Field(...,description="number of duplicate rows in the dataframe")
    correlation:str = Field(...,description="info about the correlation relationships among the features.")
    feature_data_types:str = Field(...,description="info about the intrepretation of data types of each feature.")
    summarize_statistics:str = Field(...,description="info about the summarization of statistics of the features.")

@st.cache_resource()
class DataframeAgent:
     def __init__(self,data):
        self.df = data
        self.DataframeInfoRetrive = DataframeInfoRetrive
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",temperature=0.2)


     def get_prompt_template(self):
         prompt_template = ChatPromptTemplate.from_messages([
               ("system","You are a pandas dataframe agent. \
               Read the given dataframe and give the reponse in the given output format instructions"),
               ("user", "here is the dataframe \n {dataframe}"),
               ("user","here are the output format instructions: \n {instructions}")])
         return prompt_template

     def initialize_output_parser(self):
         pydantic_output_parser = PydanticOutputParser(pydantic_object=self.DataframeInfoRetrive)
         format_instructions = pydantic_output_parser.get_format_instructions()
         return (pydantic_output_parser, format_instructions)
     

     def run_chain(self):
         df = self.df
         prompt_template = self.get_prompt_template()
         output_parser,format_instructions = self.initialize_output_parser()

         input_grid = {"dataframe":df,"instructions":format_instructions}

         chain = prompt_template | self.llm | output_parser
         response = chain.invoke(input=input_grid)

         return response
