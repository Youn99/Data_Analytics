from azure.storage.blob import ContainerClient
import json
from io import StringIO
import pandas as pd

# here we put fname_blob and not year because the files we had, they had year as the name.
def worldhappiness(fname_blob):
    conn_str='Connection_string'
    container_name= 'container_name'
    container_client = ContainerClient.from_connection_string(conn_str=conn_str, container_name=container_name)
    existing_blob_list = [blob.name for blob in container_client.list_blobs()]
    fname_blob = f'{fname_blob}.csv'
    if fname_blob in existing_blob_list:
        downloaded_blob = container_client.download_blob(fname_blob)
        df = pd.read_csv(StringIO(downloaded_blob.content_as_text()))
        df = df.to_dict('records')
        dic = {fname_blob[:4] : df}
        dump = json.dumps(dic, indent = 4)
        return dump
    else:
            return( f'Attention {fname_blob} not exists in the list, the existing files are :{existing_blob_list}')





