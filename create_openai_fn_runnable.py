import streamlit as st
from typing import Optional
from langchain.chains.structured_output import create_openai_fn_runnable
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field

# Define the data model for recording vehicle information
class RecordVehicle(BaseModel):
    '''Record some identifying information about a vehicle.'''

    make: str = Field(..., description="The vehicle's make (brand)")
    model: str = Field(..., description="The vehicle's model")
    year: int = Field(..., description="The vehicle's manufacturing year")
    color: Optional[str] = Field(None, description="The vehicle's color")

# Define the data model for recording building information
class RecordBuilding(BaseModel):
    '''Record some identifying information about a building.'''

    name: str = Field(..., description="The building's name")
    location: str = Field(..., description="The building's location")
    height: Optional[float] = Field(None, description="The building's height in meters")
    year_built: Optional[int] = Field(None, description="The year the building was constructed")

def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()
    
# Replace with your OpenAI API key
api_key_path = 'api_key.txt'
api_key = load_api_key(api_key_path)

# Initialize the OpenAI LLM with your API key and model
llm = ChatOpenAI(api_key=api_key, model="gpt-4", temperature=0)

# Create the structured output handler
structured_llm = create_openai_fn_runnable([RecordVehicle, RecordBuilding], llm)

# Streamlit UI setup
st.title("Structured Information Extractor")

# Input description from the user
description = st.text_area("Enter a description of a vehicle or a building:")

# Process the input and get structured data
if st.button("🛠️ Extract Information"):
    if description:
        try:
            result = structured_llm.invoke(description)
            st.write(result)  # Display the structured output Text
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.error("⚠️ Please enter a description before extracting information.")

# & py -3 -m streamlit run "C:\AIDI\SEM2\KNOWLEDGE AND EXP\ASSIGNMENT2\create_openai_fn_runnable.py"