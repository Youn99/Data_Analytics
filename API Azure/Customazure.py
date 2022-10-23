from azure.storage.blob import ContainerClient
import json
from io import StringIO
import pandas as pd
def humanfreedom(year):
    conn_str='<your_connection_string>'
    container_name= 'container_name'
    container_client = ContainerClient.from_connection_string(conn_str=conn_str, container_name=container_name)
    fname_blob = 'file_name'
    downloaded_blob = container_client.download_blob(fname_blob) 
    wrd_happiness = pd.read_csv(StringIO(downloaded_blob.content_as_text()))
    wrd_happ = wrd_happiness[wrd_happiness.year == int(year)]
    wrd_happ = wrd_happ.to_dict('records')
    dic = {year : wrd_happ}
    dump = json.dumps(dic, indent = 4)
    return dump