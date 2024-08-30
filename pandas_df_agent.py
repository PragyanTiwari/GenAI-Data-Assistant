
import json
import streamlit as st

from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel,Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser

# importing the field descriptions
with open("prompt.json","r") as prompt_file:
    prompt = json.load(prompt_file)['df_info_retrieve_desc']


# Building the Pydantic JSON Schema for the output parser
class DataframeInfoRetrive(BaseModel):
    """output response class"""
    feature_Explanation:str = Field(...,description=prompt['feature_Explanation'])
    shape:tuple = Field(...,description=prompt['shape'])
    n_duplicates:int = Field(...,description=prompt['n_duplicates'])
    correlation:str = Field(...,description=prompt['correlation'])
    feature_data_types:str = Field(...,description=prompt['feature_data_types'])
    summarize_statistics:str = Field(...,description=prompt['summarize_statistics'])



@st.cache_resource()
class DataframeAgent:
     def __init__(self,data):
        self.df = data
        self.DataframeInfoRetrive = DataframeInfoRetrive
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",temperature=0.2)



     def get_prompt_template(self):
         """
         message prompt template with placeholders: dataframe & instructions
         """
         prompt_template = ChatPromptTemplate.from_messages([
               ("system","You are a pandas dataframe agent. \
               Read the given dataframe and give the reponse in the given output format instructions"),
               ("user", "here is the dataframe \n {dataframe}"),
               ("user","here are the output format instructions: \n {instructions}")])
         return prompt_template



     def initialize_output_parser(self):
         """
         output parser with output format instructions. 
         Pydantic Schema is used for the output parser.
         """
         pydantic_output_parser = PydanticOutputParser(pydantic_object=self.DataframeInfoRetrive)
         format_instructions = pydantic_output_parser.get_format_instructions()
         return (pydantic_output_parser, format_instructions)
     


     def run_chain(self):
         """
         LCEL Chain for the Agent.
         prompt template -> llm -> output parser
         """
         df = self.df
         prompt_template = self.get_prompt_template()
         output_parser,format_instructions = self.initialize_output_parser()

         input_grid = {"dataframe":df,"instructions":format_instructions}

         chain = prompt_template | self.llm | output_parser
         response = chain.invoke(input=input_grid)

         return response
