# Databricks notebook source
# MAGIC %pip install Pdfplumber langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os 
from pyspark.sql.functions import substring_index

dir_path = '/Volumes/workspace/default/rag_pdfs'  # Adjust to '/dbfs/' to access DBFS
file_paths = [file.path for file in dbutils.fs.ls(dir_path)]

df = spark.createDataFrame(file_paths, 'string').select(substring_index("value","/",-1).alias('file_name'))

df.show()

# COMMAND ----------

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

pdf_volume_dir = '/Volumes/workspace/default/rag_pdfs'

processed_files = spark.sql(f"SELECT DISTINCT file_name FROM doc_track").collect()
processed_files = set(row['file_name'] for row in processed_files)

new_files =[file for file in os.listdir(pdf_volume_dir) if file not in processed_files]

all_text = ''

for file_name in new_files:
    pdf_path = os.path.join(pdf_volume_dir, file_name)
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            single_page_text = page.extract_text()
            all_text = all_text +'\n'+single_page_text

# COMMAND ----------

all_text

# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter
len_funtion = len

splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                          chunk_overlap=100,
                                          length_function=len_funtion)

chunks = splitter.split_text(all_text)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType,StringType
import pandas as pd 

@pandas_udf("array<string>")
def get_chunks(dummy):
    return pd.Series([chunks])

spark.udf.register("get_chunks", get_chunks)

# COMMAND ----------

spark.sql("""
insert into rag_pdfsdocs_text(text)
select explode(get_chunks('dummy')) as text;
""")

# COMMAND ----------

# Read the table into a DataFrame
df = spark.read.table("rag_pdfsdocs_text")

# Show the DataFrame content
print(df.count())

df.show(truncate=False)

