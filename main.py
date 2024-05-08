import db_connector
import getdata
import model

# Create a connection object
conn = db_connector.connect()

# Get data
df = getdata.getdata_datastore_license_curr(conn)
print(df)

# Call clustering algorithm from model.py
clusters = model.clustering_algorithm(df)

# test ML model using test data








