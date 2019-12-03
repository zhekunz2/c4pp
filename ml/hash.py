from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
import numpy

# Dimension of our vector space
dimension = 500

# Create a random binary hash with 10 bits
rbp = RandomBinaryProjections('rbp', 10)

# Create engine with pipeline configuration
engine = Engine(dimension, lshashes=[rbp])

# Index 1000000 random vectors (set their data to a unique string)
for index in range(100000):
    v = numpy.random.randn(dimension)
    engine.store_vector(v, 'data_%d' % index)

# Create random query vector
query = numpy.random.randn(dimension)

# Get nearest neighbours
N = engine.neighbours(query)
for lshash in engine.lshashes:
    for bucket_key in lshash.hash_vector(v, querying=True):
        bucket_content = engine.storage.get_bucket(lshash.hash_name,
                                                 bucket_key)
        print(bucket_key)
        print(len(bucket_content))
