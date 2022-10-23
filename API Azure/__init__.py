import logging
import azure.functions as func 
from azure.storage.blob import ContainerClient
import sys
import os
from sys import path
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path)
from Customazure import humanfreedom

def main(req: func.HttpRequest) -> func.HttpResponse:

    logging.info('Python HTTP trigger function processed a request.')
    year = req.params.get('year')
    if not year:

        return func.HttpResponse(

            "Insert the year please.",

            status_code=200

        )

    if year:
        result = humanfreedom(year)
        return func.HttpResponse(result, status_code=200)