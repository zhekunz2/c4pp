#!/usr/bin/env python
import azure.batch.batch_service_client as batch
import azure.batch.batch_auth as batchauth

_BATCH_ACCOUNT_NAME = 'batch3'
_BATCH_ACCOUNT_KEY = 'X06WnHaOyoTogV90Jbr4/KHpTFEe3LSSzxts7wTeM6GVpA2TwL0ALxpGnW/lv/kdEK3ayXt8R/OOTxpghivJjA=='
_BATCH_ACCOUNT_URL = 'https://batch3.centralus.batch.azure.com'
_JOB_ID = 'job0318'
_POOL_ID = 'pool0318'
credentials = batchauth.SharedKeyCredentials(_BATCH_ACCOUNT_NAME, _BATCH_ACCOUNT_KEY)

batch_client = batch.BatchServiceClient(
        credentials,
        batch_url=_BATCH_ACCOUNT_URL)
batch_client.job.delete(_JOB_ID)
batch_client.pool.delete(_POOL_ID)
