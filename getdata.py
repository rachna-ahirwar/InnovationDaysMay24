# fetch data from hive db
import pandas as pd

def getdata_datastore_license_curr(conn):

    # Create a cursor object
    cursor = conn.cursor()

    # Execute a query to select data from table - innovation
    cursor.execute("SELECT * FROM innovation LIMIT 100")

    # Load the data into a DataFrame
    df1 = pd.DataFrame(cursor.fetchall())
    #print(df1)

    # Close the cursor and connection
    cursor.close()
    conn.close()

    # Combine the data from the different tables into a single DataFrame
    #df = pd.concat([df1, df2], axis=0)

    # Print the DataFrame
    # print(df)
    # print(df.columns)

    return df1
