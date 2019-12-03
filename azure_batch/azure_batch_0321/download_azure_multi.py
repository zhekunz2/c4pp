#!/usr/bin/env python
from azure.storage.blob import BlockBlobService
from azure.storage.blob import PublicAccess
import os
import sys
from Queue import Queue
from threading import Thread

#name of your storage account and the access key from Settings->AccessKeys->key1
block_blob_service = BlockBlobService(account_name='c4pp', account_key='N68nxGso5C2RrLNyZHpW+kiipWMHbtsmLrTmRjWF16tAscXTlWVAqBjFMJAcCp5Ue6Zgm9Y7nMPeayA0A1W/lQ==')

#name of the container
container_name = 'mm1out'
generator = block_blob_service.list_blobs(container_name)

#code below lists all the blobs in the container and downloads them one after another

def do_stuff(q):
  while True:
    name = q.get()
    print(name)
    block_blob_service.get_blob_to_path(container_name,name,name)
    q.task_done()

q = Queue(maxsize=0)
num_threads = 12

for i in range(num_threads):
  worker = Thread(target=do_stuff, args=(q,))
  worker.setDaemon(True)
  worker.start()

#for x in range(100):
#  q.put(x)
for blob in generator:
    #if "metric" in blob.name or "_rt_" in blob.name:
    if "tar.gz" in blob.name:
	q.put(blob.name)
        #print(blob.name)


q.join()
    #print("{}".format(blob.name))
    #check if the path contains a folder structure, create the folder structure
    #if "/" in "{}".format(blob.name):
    #    print("there is a path in this")
    #    #extract the folder path and check if that folder exists locally, and if not create it
    #    head, tail = os.path.split("{}".format(blob.name))
    #    print(head)
    #    print(tail)
    #    if (os.path.isdir(os.getcwd()+ "/" + head)):
    #        #download the files to this directory
    #        print("directory and sub directories exist")
    #        block_blob_service.get_blob_to_path(container_name,blob.name,os.getcwd()+ "/" + head + "/" + tail)
    #    else:
    #        #create the diretcory and download the file to it
    #        print("directory doesn't exist, creating it now")
    #        os.makedirs(os.getcwd()+ "/" + head, exist_ok=True)
    #        print("directory created, download initiated")
    #        block_blob_service.get_blob_to_path(container_name,blob.name,os.getcwd()+ "/" + head + "/" + tail)
    #else:
#print(sys.argv[1])
#block_blob_service.get_blob_to_path(container_name,sys.argv[1],sys.argv[1])

