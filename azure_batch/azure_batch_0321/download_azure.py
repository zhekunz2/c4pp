#!/usr/bin/env python
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess
import os

#name of your storage account and the access key from Settings->AccessKeys->key1
block_blob_service = BlockBlobService(account_name='c4pp', account_key='N68nxGso5C2RrLNyZHpW+kiipWMHbtsmLrTmRjWF16tAscXTlWVAqBjFMJAcCp5Ue6Zgm9Y7nMPeayA0A1W/lQ==')

#name of the container
container_name = 'lrm31out'
generator = block_blob_service.list_blobs(container_name)

#code below lists all the blobs in the container and downloads them one after another
for blob in generator:
    print(blob.name)
    print("{}".format(blob.name))
    #check if the path contains a folder structure, create the folder structure
    if "/" in "{}".format(blob.name):
        print("there is a path in this")
        #extract the folder path and check if that folder exists locally, and if not create it
        head, tail = os.path.split("{}".format(blob.name))
        print(head)
        print(tail)
        if (os.path.isdir(os.getcwd()+ "/" + head)):
            #download the files to this directory
            print("directory and sub directories exist")
            block_blob_service.get_blob_to_path(container_name,blob.name,os.getcwd()+ "/" + head + "/" + tail)
        else:
            #create the diretcory and download the file to it
            print("directory doesn't exist, creating it now")
            os.makedirs(os.getcwd()+ "/" + head, exist_ok=True)
            print("directory created, download initiated")
            block_blob_service.get_blob_to_path(container_name,blob.name,os.getcwd()+ "/" + head + "/" + tail)
    else:
        block_blob_service.get_blob_to_path(container_name,blob.name,blob.name)

