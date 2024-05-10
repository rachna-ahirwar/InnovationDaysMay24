import db_connector
import getdata
import model

# Create a connection object
conn = db_connector.connect()

# Get data
# df = getdata.getdata_datastore_license_curr(conn)
# df.dropna(inplace=True)
# # df = df.iloc[:, 1:]
# df.to_csv('file3.csv', index=False)

#print(df)

# Call clustering algorithm from model.py
model.clustering_algorithm("file4.csv")
#model.clustering_algorithm("file2.csv")
#model.gaussianMixture("file2.csv")

# test ML model using test data








