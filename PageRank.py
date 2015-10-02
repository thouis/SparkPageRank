import pyspark


def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)

def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def push_to_neighbors(neighbor_list,
                      weight,
                      fraction):
    num_neighbors = len(neighbor_list)
    w = weight * fraction / num_neighbors
    return [(n, w) for n in neighbor_list]


def pagerank(neighbors,
             page_names,
             iterations=10,
             inspect=[]):
    ''' run the pagerank algorithm'''

    give_fraction = 0.85
    keep_fraction = 0.15

    # set each node to rank 1.0
    page_ranks = neighbors.mapValues(lambda _: 1.0)

    # neighbors and page_ranks are now copartitioned.  We want this to always
    # be true!

    for i in range(iterations):
        # bring weights and neighbors together.
        # First make sure they are copartitioned.
        assert copartitioned(neighbors, page_ranks)
        neighbors_join_ranks = neighbors.join(page_ranks)

        # push weights out to neighbors
        out_ranks = neighbors_join_ranks.flatMapValues(
            lambda (neighbor_list, rank): push_to_neighbors(neighbor_list,
                                                            rank, give_fraction))

        # We used flatMapValues, but that leaves the key in place.  We just
        # want the values.
        out_ranks = out_ranks.values()

        # sum contribution to each node.  Use the same number of partitions as
        # neighbors, so they end up copartitioned.
        summed_ranks = out_ranks.reduceByKey(lambda x, y: x + y,
                                             numPartitions=neighbors.getNumPartitions())
        assert copartitioned(summed_ranks, neighbors)

        # Add in keep_fraction, and update.
        # Cache because this is the only RDD that is used in the next iteration.
        page_ranks = summed_ranks.mapValues(lambda x: x + keep_fraction).cache()

        print("Iteration {0}".format(i))
        for node in inspect:
            print("   {0}: {1}".format(page_names.lookup(node)[0],
                                       page_ranks.lookup(node)[0]))

if __name__ == '__main__':
    sc = pyspark.SparkContext()
    sc.setLogLevel('WARN')

    links = sc.textFile('s3://Harvard-CS205/wikipedia/links-simple-sorted.txt', 32)
    page_names = sc.textFile('s3://Harvard-CS205/wikipedia/titles-sorted.txt', 32)

    # process links into (node #, [neighbor node #, neighbor node #, ...]
    neighbor_graph = links.map(link_string_to_KV)

    # create an RDD for looking up page names from numbers
    # remember that it's all 1-indexed
    page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n)).sortByKey().cache()
    # set up partitioning - we have roughly 8 workers, if we're on AWS with 2
    # nodes not counting the driver.  This is 4 partitions per worker.
    # Cache this result, so we don't recompute the link_string_to_KV() each time.
    neighbor_graph = neighbor_graph.partitionBy(256).cache()

    # find Kevin Bacon
    Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
    assert len(Kevin_Bacon) == 1
    Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

    # find Harvard University
    Harvard_University = page_names.filter(lambda (K, V): V == 'Harvard_University').collect()
    assert len(Harvard_University) == 1
    Harvard_University = Harvard_University[0][0]  # extract node id

    pagerank(neighbor_graph,
             page_names,
             iterations=10,
             inspect=[Kevin_Bacon, Harvard_University])


#  aws emr create-cluster --name "Spark cluster" --release-label emr-4.1.0 --applications Name=Spark --ec2-attributes KeyName=SparkKeyPair --instance-type m3.xlarge --instance-count 5 --use-default-roles
