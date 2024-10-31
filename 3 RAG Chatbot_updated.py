# Databricks notebook source
# DBTITLE 1,Install Dependencies
# MAGIC %pip install mlflow==2.14.3 langchain==0.1.5 databricks-vectorsearch==0.22 databricks-sdk==0.18.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Set needed parameters
import os

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get(scope ="access-token", key = "my-token")

index_name="llm.rag.docs_idx"
#host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

VECTOR_SEARCH_ENDPOINT_NAME="grc_vector_search"

# COMMAND ----------

# DBTITLE 1,Build Retriever
from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings

embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

def get_retriever(persist_dir: str = None):
    os.environ["DATABRICKS_HOST"] = host
    #Get the vector search index
    vsc = VectorSearchClient(workspace_url=host, personal_access_token=os.environ["DATABRICKS_TOKEN"])
    vs_index = vsc.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="text", embedding=embedding_model
    )
    return vectorstore.as_retriever()



# COMMAND ----------

# DBTITLE 1,Create the RAG Langchain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

TEMPLATE = """### Instruction ###
You are an AI language model designed to function as a specialized Retrieval-Augmented Generation (RAG) assistant. Your role is to assist users with questions specifically related to ISO 27001 Information Security Management System and ISO 42001 AI Management System, using information retrieved directly from the provided uploaded documents. The document context, including relevant clauses and sections, will be provided in {context}.

When generating responses, prioritize correctness and ensure that answers are:
Grounded in the specific context of the uploaded documents, with clear references to clauses, controls, and policies (including their numbers) where applicable.
Accurate, relevant, and detailed, closely following the structure and terminology used in the documents, while ensuring responses are clear and accessible to all audiences.
For each user query:
Cite the relevant ISO 27001 or 42001 clause, policy, or control number that applies, explaining it in detail for full understanding.

Where necessary, add context or reasoning to clarify how the specific parts of the standards apply. Avoid redundancy but expand on key points to ensure thorough understanding. All answers should be well-aligned with the exact provisions and requirements laid out in the ISO standards in the uploaded documents.
### Context ###
{context}
### Input ###
{question}
### Output ###

"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)



# COMMAND ----------

# DBTITLE 1,Test Langchain
question = {"query": "How does ISO 42001 address the issue of AI bias?"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

# DBTITLE 1,Register our Chain as a model to Unity Catalog
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = "llm.rag.chatbotv2"

with mlflow.start_run(run_name="appliance_chatbot_run") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,  # Load the retriever with DATABRICKS_TOKEN env as secret (for authentication).
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )
