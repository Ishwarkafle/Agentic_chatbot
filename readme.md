conda create -n LangGraph python=3.13 -y

conda activate LangGraph

pip install -r requirement.txt

from langchain_groq import ChatGroq
