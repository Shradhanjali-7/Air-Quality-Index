import pandas as pd
from pymongo import MongoClient
import json
# Load the file
df = pd.read_csv("PRSA_Data_Aotizhongxin_20130301-20170228")
df = pd.read_csv("PRSA_Data_Changping_20130301-20170228")
df = pd.read_csv("PRSA_Data_Dingling_20130301-20170228")
df = pd.read_csv("PRSA_Data_Dongsi_20130301-20170228")
df = pd.read_csv("PRSA_Data_Guanyuan_20130301-20170228")
df = pd.read_csv("PRSA_Data_Gucheng_20130301-20170228")
df = pd.read_csv("PRSA_Data_Huairou_20130301-20170228")
df = pd.read_csv("PRSA_Data_Nongzhanguan_20130301-20170228")
df = pd.read_csv("PRSA_Data_Shunyi_20130301-20170228")
df = pd.read_csv("PRSA_Data_Tiantan_20130301-20170228")
df = pd.read_csv("PRSA_Data_Wanliu_20130301-20170228")
df = pd.read_csv("PRSA_Data_Wanshouxigong_20130301-20170228")
df.dropna(inplace=True)
# Create 'time' column
df['time'] = pd.to_datetime(df[['year', 'month', 'day',
'hour']])
# Drop split time columns, but keep station name
df = df.drop(columns=['No', 'year', 'month', 'day', 'hour'])
# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["AQI"]
collection = db["china_data"]
#collection.delete_many({}) # Clear previous runs
# Insert into MongoDB
#data_json = json.loads(df.to_json(orient='records'))
#collection.insert_many(data_json)
#print(f"Inserted {len(data_json)} records into MongoDB.")
