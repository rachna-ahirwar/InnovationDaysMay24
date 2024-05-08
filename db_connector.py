from pyhive import hive

# Create a connection object
def connect():
    conn = hive.Connection(host='cxo-hdp-prod4.cxo.storage.hpecorp.net', 
                        port=10000, 
                        username='rachna', 
                        database='ahirwarr')
    return conn
