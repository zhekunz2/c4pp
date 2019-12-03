#!/usr/bin/python
from __future__ import print_function
import datetime
import io
import os
import sys
import time

try:
    input = raw_input
except NameError:
    pass

import azure.storage.blob as azureblob
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batchauth
import azure.batch.models as batchmodels

sys.path.append('.')
sys.path.append('..')


# Update the Batch and Storage account credential strings below with the values
# unique to your accounts. These are used when constructing connection strings
# for the Batch and Storage client objects.

# global
_BATCH_ACCOUNT_NAME = 'batch3'
_BATCH_ACCOUNT_KEY = 'bs9rp3gfYXCgXgLiLkrna9HTd04W/uEeZnM5D8gKfyhOSwfO9KcxjL8xYsXG+2t9eagteIRO50t4njbJHva7OA=='
_BATCH_ACCOUNT_URL = 'https://batch3.centralus.batch.azure.com'
_STORAGE_ACCOUNT_NAME = 'c4pp'
_STORAGE_ACCOUNT_KEY = 'N68nxGso5C2RrLNyZHpW+kiipWMHbtsmLrTmRjWF16tAscXTlWVAqBjFMJAcCp5Ue6Zgm9Y7nMPeayA0A1W/lQ=='
_POOL_ID = 'pool0318'
_DEDICATED_POOL_NODE_COUNT = 2
_LOW_PRIORITY_POOL_NODE_COUNT = 0
_POOL_VM_SIZE = 'STANDARD_D3_v2'
_JOB_ID = 'job0318'
_MAX_TASKS_PER_NODE = 4

input_dir = 'restlrm2'
resource_dir = 'stanresources'
input_container_name = input_dir
output_container_name = 'lrm2out'
batch_size = range(21, 31)
to_create_pool = True

def query_yes_no(question, default="yes"):
    """
    Prompts the user for yes/no input, displaying the specified question text.

    :param str question: The text of the prompt for input.
    :param str default: The default if the user hits <ENTER>. Acceptable values
    are 'yes', 'no', and None.
    :rtype: str
    :return: 'yes' or 'no'
    """
    valid = {'y': 'yes', 'n': 'no'}
    if default is None:
        prompt = ' [y/n] '
    elif default == 'yes':
        prompt = ' [Y/n] '
    elif default == 'no':
        prompt = ' [y/N] '
    else:
        raise ValueError("Invalid default answer: '{}'".format(default))

    while 1:
        choice = input(question + prompt).lower()
        if default and not choice:
            return default
        try:
            return valid[choice[0]]
        except (KeyError, IndexError):
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def print_batch_exception(batch_exception):
    """
    Prints the contents of the specified Batch exception.

    :param batch_exception:
    """
    print('-------------------------------------------')
    print('Exception encountered:')
    if batch_exception.error and \
            batch_exception.error.message and \
            batch_exception.error.message.value:
        print(batch_exception.error.message.value)
        if batch_exception.error.values:
            print()
            for mesg in batch_exception.error.values:
                print('{}:\t{}'.format(mesg.key, mesg.value))
    print('-------------------------------------------')


def upload_file_to_container(block_blob_client, container_name, file_path):
    """
    Uploads a local file to an Azure Blob storage container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param str file_path: The local path to the file.
    :rtype: `azure.batch.models.ResourceFile`
    :return: A ResourceFile initialized with a SAS URL appropriate for Batch
    tasks.
    """
    blob_name = os.path.basename(file_path)

    print('Uploading file {} to container [{}]...'.format(file_path,
                                                          container_name))

    block_blob_client.create_blob_from_path(container_name,
                                            blob_name,
                                            file_path)

    # Obtain the SAS token for the container.
    sas_token = get_container_sas_token(block_blob_client,
                            container_name, azureblob.BlobPermissions.READ)


    sas_url = block_blob_client.make_blob_url(container_name,
                                              blob_name,
                                              sas_token=sas_token)

    return batchmodels.ResourceFile(file_path=blob_name,
                                    blob_source=sas_url)

def get_container_sas_token(block_blob_client,
                            container_name, blob_permissions):
    """
    Obtains a shared access signature granting the specified permissions to the
    container.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param BlobPermissions blob_permissions:
    :rtype: str
    :return: A SAS token granting the specified permissions to the container.
    """
    # Obtain the SAS token for the container, setting the expiry time and
    # permissions. In this case, no start time is specified, so the shared
    # access signature becomes valid immediately. Expiration is in 2 hours.
    container_sas_token = \
        block_blob_client.generate_container_shared_access_signature(
            container_name,
            permission=blob_permissions,
            expiry=datetime.datetime.utcnow() + datetime.timedelta(hours=365))

    return container_sas_token



def get_container_sas_url(block_blob_client,
                            container_name, blob_permissions):
    """
    Obtains a shared access signature URL that provides write access to the 
    ouput container to which the tasks will upload their output.

    :param block_blob_client: A blob service client.
    :type block_blob_client: `azure.storage.blob.BlockBlobService`
    :param str container_name: The name of the Azure Blob storage container.
    :param BlobPermissions blob_permissions:
    :rtype: str
    :return: A SAS URL granting the specified permissions to the container.
    """
    # Obtain the SAS token for the container.
    sas_token = get_container_sas_token(block_blob_client,
                            container_name, azureblob.BlobPermissions.WRITE)

    # Construct SAS URL for the container
    container_sas_url = "https://{}.blob.core.windows.net/{}?{}".format(_STORAGE_ACCOUNT_NAME, container_name, sas_token)

    return container_sas_url


def create_pool(batch_service_client, pool_id, users):
    """
    Creates a pool of compute nodes with the specified OS settings.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str pool_id: An ID for the new pool.
    :param str publisher: Marketplace image publisher
    :param str offer: Marketplace image offer
    :param str sku: Marketplace image sky
    """
    print('Creating pool [{}]...'.format(pool_id))

    # Create a new pool of Linux compute nodes using an Azure Virtual Machines
    # Marketplace image. For more information about creating pools of Linux
    # nodes, see:
    # https://azure.microsoft.com/documentation/articles/batch-linux-nodes/

    # The start task installs ffmpeg on each node from an available repository, using
    # an administrator user identity.

    new_pool = batch.models.PoolAddParameter(
        id=pool_id,
        user_accounts=users,
	virtual_machine_configuration=batchmodels.VirtualMachineConfiguration(
        image_reference=batchmodels.ImageReference(
            publisher="Canonical",
            offer="UbuntuServer",
            sku="18.04-LTS",
            version="latest"
            ),
        node_agent_sku_id="batch.node.ubuntu 18.04"),
        vm_size=_POOL_VM_SIZE,
        target_dedicated_nodes=_DEDICATED_POOL_NODE_COUNT,
        target_low_priority_nodes=_LOW_PRIORITY_POOL_NODE_COUNT,
        start_task=batchmodels.StartTask(
            #command_line="/bin/bash -c \"git clone https://github.com/uiuc-arc/probfuzz.git; \
            #        cd probfuzz/; ./install_java.sh; ./install.sh\"",
            # command_line="/bin/bash -c \"apt-get update\"",
            command_line = "/bin/bash -c \"sudo apt-get -y update; \
                sudo apt-get install git; \
                sudo apt-get install -y python2.7; \
                sudo apt-get install -y python-pip; \
                sudo apt-get install -y bc; \
                sudo apt-get install -y r-base; \
                sudo pip2 --no-cache-dir install pandas; \
                sudo pip2 install rpy2==2.8.6;\
                sudo pip2 install argparse;\
                sudo pip2 install numpy;\
                sudo pip2 install scipy;\
            \"",
            wait_for_success=True,
            user_identity=batchmodels.UserIdentity(user_name="admin"),
            #user_identity=batchmodels.UserIdentity(
            #    auto_user=batchmodels.AutoUserSpecification(
            #    scope=batchmodels.AutoUserScope.pool,
            #    elevation_level=batchmodels.ElevationLevel.admin)),
         ),
        max_tasks_per_node=_MAX_TASKS_PER_NODE
    )
    batch_service_client.pool.add(new_pool)

def create_job(batch_service_client, job_id, resource_files, pool_id):
    """
    Creates a job with the specified ID, associated with the specified pool.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID for the job.
    :param str pool_id: The ID for the pool.
    """
    print('Creating job [{}]...'.format(job_id))

    job = batch.models.JobAddParameter(
        id=job_id,
        pool_info=batch.models.PoolInformation(pool_id=pool_id),
        job_preparation_task=batchmodels.JobPreparationTask(
            command_line = "/bin/bash -c \"sudo apt-get -y update; \
                sudo chmod o+w /usr/local/lib/R/site-library; \
                ./installrpy.sh; \
                tar -xf cmdstan.tar.gz; \
                cd cmdstan; \
                make build; \
                cd -; \
            \"",
            wait_for_success=True,
            resource_files=resource_files,
            user_identity=batchmodels.UserIdentity(user_name="admin"),
            )
        )

    batch_service_client.job.add(job)

def add_tasks(batch_service_client, job_id, input_files, output_container_sas_url):
    """
    Adds a task for each input file in the collection to the specified job.

    :param batch_service_client: A Batch service client.
    :type batch_service_client: `azure.batch.BatchServiceClient`
    :param str job_id: The ID of the job to which to add the tasks.
    :param list input_files: A collection of input files. One task will be
     created for each input file.
    :param output_container_sas_token: A SAS token granting write access to
    the specified Azure Blob storage container.
    """

    #print('Adding {} tasks to job [{}]...'.format(len(input_files), job_id))
    print('Adding {} tasks to job [{}]...'.format(_LOW_PRIORITY_POOL_NODE_COUNT + _DEDICATED_POOL_NODE_COUNT, job_id))

    tasks = list()

    for idx, input_file in enumerate(input_files):
        if idx >= 8:
            break
        input_file_path=input_file.file_path
        input_file_name=input_file_path.split('/')[-1].split('.')[0]
        #output_file_path="".join((input_file_path).split('.')[:-2]) + 'metrics_out_0319' + '.txt'
        # TODO: cp task to stan dir; unzip; build; change mnt script; upload metrics & summary to storage
        command = "/bin/bash -c \" \
                time ./tar_to_metrics.sh {0} &> {1}_log.txt; \
        \"".format(input_file_path, input_file_name)
                #
                #./run_metrics.sh ./metrics.py /mnt/batch/tasks/workitems/RunByIterJobAll/job-1/Task*/wd/example-models/; \
                #./runner_10.sh /mnt/batch/tasks/workitems/RunByIterJobAll/job-1/Task*/wd/example-models/ _*.csv; \
                #pip2 install --upgrade pip2; \
                #sudo apt-get install -y r-base-core; \
        #command = "/bin/bash -c \"touch probfuzz.txt\""
        tasks.append(batch.models.TaskAddParameter(
            id='Task_{}'.format(input_file_name),
            command_line=command,
            user_identity=batchmodels.UserIdentity(user_name="admin"),
            resource_files=input_files,
            output_files=[batchmodels.OutputFile(
                      file_pattern="*.txt".format(idx),
                      destination=batchmodels.OutputFileDestination(
                        container=batchmodels.OutputFileBlobContainerDestination(
                            container_url=output_container_sas_url)),
                      upload_options=batchmodels.OutputFileUploadOptions(
                        upload_condition=batchmodels.OutputFileUploadCondition.task_success))]
        ))
    batch_service_client.task.add_collection(job_id, tasks)


# def wait_for_tasks_to_complete(batch_service_client, job_id, timeout):
#     """
#     Returns when all tasks in the specified job reach the Completed state.
# 
#     :param batch_service_client: A Batch service client.
#     :type batch_service_client: `azure.batch.BatchServiceClient`
#     :param str job_id: The id of the job whose tasks should be monitored.
#     :param timedelta timeout: The duration to wait for task completion. If all
#     tasks in the specified job do not reach Completed state within this time
#     period, an exception will be raised.
#     """
#     timeout_expiration = datetime.datetime.now() + timeout
# 
#     print("Monitoring all tasks for 'Completed' state, timeout in {}..."
#           .format(timeout), end='')
# 
#     while datetime.datetime.now() < timeout_expiration:
#         print('.', end='')
#         sys.stdout.flush()
#         tasks = batch_service_client.task.list(job_id)
# 
#         incomplete_tasks = [task for task in tasks if
#                             task.state != batchmodels.TaskState.completed]
#         if not incomplete_tasks:
#             print()
#             return True
#         else:
#             time.sleep(1)
#
#     print()
#     raise RuntimeError("ERROR: Tasks did not reach 'Completed' state within "
#                        "timeout period of " + str(timeout))



if __name__ == '__main__':

    start_time = datetime.datetime.now().replace(microsecond=0)
    print('Sample start: {}'.format(start_time))

    # Create the blob client, for use in obtaining references to
    # blob storage containers and uploading files to containers.


    blob_client = azureblob.BlockBlobService(
        account_name=_STORAGE_ACCOUNT_NAME,
        account_key=_STORAGE_ACCOUNT_KEY)

    # Use the blob client to create the containers in Azure Storage if they
    # don't yet exist.

    blob_client.create_container(input_container_name, fail_on_exist=False)
    blob_client.create_container(output_container_name, fail_on_exist=False)
    print('Container [{}] created.'.format(input_container_name))
    print('Container [{}] created.'.format(output_container_name))

    input_file_paths = []

    for folder, subs, files in os.walk(os.path.join(sys.path[0],input_dir)):
        for filename in files:
            if filename.endswith(".tar.gz"):
                input_file_paths.append(os.path.abspath(os.path.join(folder, filename)))

     #Upload the input files. This is the collection of files that are to be processed by the tasks. 
    input_files = [
         upload_file_to_container(blob_client, input_container_name, file_path)
         for file_path in input_file_paths]
    #input_files = []

    resource_file_paths = []
    for folder, subs, files in os.walk(os.path.join(sys.path[0],resource_dir)):
        for filename in files:
            resource_file_paths.append(os.path.abspath(os.path.join(folder, filename)))

     #Upload the resource files. Used when starting a job
    resource_files = [
         upload_file_to_container(blob_client, input_container_name, file_path)
         for file_path in resource_file_paths]
    #input_files = []
    # Obtain a shared access signature URL that provides write access to the output
    # container to which the tasks will upload their output.

    output_container_sas_url = get_container_sas_url(
        blob_client,
        output_container_name,
        azureblob.BlobPermissions.WRITE)

    # Create a Batch service client. We'll now be interacting with the Batch
    # service in addition to Storage
    credentials = batchauth.SharedKeyCredentials(_BATCH_ACCOUNT_NAME,
                                                 _BATCH_ACCOUNT_KEY)

    batch_client = batch.BatchServiceClient(
        credentials,
        batch_url=_BATCH_ACCOUNT_URL)

    try:
        users = [
            batchmodels.UserAccount(
            name = 'admin',
            password = 'abc123A',
            elevation_level=batchmodels.ElevationLevel.admin)
        ]
        # Create the pool that will contain the compute nodes that will execute the
        # tasks.
        if to_create_pool:
            create_pool(batch_client, _POOL_ID, users)

        # Create the job that will run the tasks.
        create_job(batch_client, _JOB_ID, resource_files,  _POOL_ID)


        # Add the tasks to the job. Pass the input files and a SAS URL 
        # to the storage container for output files.
        add_tasks(batch_client, _JOB_ID, input_files, output_container_sas_url)

        # Pause execution until tasks reach Completed state.
        #wait_for_tasks_to_complete(batch_client,
        #                       _JOB_ID,
        #                       datetime.timedelta(minutes=30))

        #print("  Success! All tasks reached the 'Completed' state within the "
        #  "specified timeout period.")

    except batchmodels.BatchErrorException as err:
        print_batch_exception(err)
        raise

    # Delete input container in storage
    # print('Deleting container [{}]...'.format(input_container_name))
    # blob_client.delete_container(input_container_name)

    # Print out some timing info
    #end_time = datetime.datetime.now().replace(microsecond=0)
    #print()
    #print('Sample end: {}'.format(end_time))
    #print('Elapsed time: {}'.format(end_time - start_time))
    #print()

    # Clean up Batch resources (if the user so chooses).
    #if query_yes_no('Delete job?') == 'yes':
    #    batch_client.job.delete(_JOB_ID)

    #if query_yes_no('Delete pool?') == 'yes':
    #    batch_client.pool.delete(_POOL_ID)

    #print()
    #input('Press ENTER to exit...')
