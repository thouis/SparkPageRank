import pyspark

def link_string_to_KV(s):
    src, dests = s.split(': ')
    dests = [int(to) for to in dests.split(' ')]
    return (int(src), dests)


def copartitioned(RDD1, RDD2):
    "check if two RDDs are copartitioned"
    return RDD1.partitioner == RDD2.partitioner


def push_to_neighbors(neighbor_list,
                      rank,
                      fraction):
    num_neighbors = len(neighbor_list)
    w = rank * fraction / num_neighbors
    return [(n, w) for n in neighbor_list]


def pagerank(neighbors,
             page_names,
             iterations=10,
             inspect=[]):
    ''' run the pagerank algorithm'''

    # PageRank has two constants that control its behavior.
    #
    # click_fraction is what percentage of the time a hypothetical random
    # surfer clicks on a link on their current page.
    #
    # jump_fraction is the percentage of the time that they jump to a random
    # page, instead.
    #
    # they should sum to 1.0
    click_fraction = 0.85
    jump_fraction = 0.15

    # set each node to rank 1.0, so all nodes equal
    # (in many implementations, this is set to 1.0 / number_of_pages)
    page_ranks = neighbors.mapValues(lambda _: 1.0)
    assert copartitioned(neighbors, page_ranks)

    #########################################################################
    # neighbors and page_ranks are now copartitioned.  We want this to always
    # be true!
    #########################################################################

    for i in range(iterations):
        # bring weights and neighbors together, using join()
        # First make sure they are copartitioned.
        assert copartitioned(neighbors, page_ranks)

        # There is no shuffle here, though there may be some communication for
        # partitions that are on different nodes.
        neighbors_join_ranks = neighbors.join(page_ranks)

        assert copartitioned(neighbors_join_ranks, neighbors)

        # neighbors_join_ranks is now:
        #     [(node, (neighbor_list, node_rank)), (node, (...)), ...]

        # Push ranks out to neighbors
        #
        # Each page with out-links divides up its current rank and hands it off
        # to its neighbors (scaled by click_fraction).
        #
        # We first call .values(), because we don't care where the current rank
        # is coming from, just where it's going to.
        out_ranks = neighbors_join_ranks.values().flatMap(
            lambda (neighbor_list, rank): push_to_neighbors(neighbor_list,
                                                            rank,
                                                            click_fraction))

        ######################################################################
        # out_ranks is not copartitioned with neighbors.  That is OK, as we're
        # not going to join it to anything.  The next step will re-partition it
        # so that it is copartitioned.
        ######################################################################

        # Sum ranks for each node.  Use the same number of partitions as
        # neighbors, so they end up copartitioned.
        summed_ranks = out_ranks.reduceByKey(
            lambda x, y: x + y, numPartitions=neighbors.getNumPartitions())

        # Make sure what we just claimed about copartitioning is true!
        assert copartitioned(summed_ranks, neighbors)

        # Add in jump_fraction, and update page_ranks.
        ######################################################################
        # Cache because this is the only RDD that is used in the next iteration.
        ######################################################################
        page_ranks = summed_ranks.mapValues(lambda x: x + jump_fraction).cache()

        # Report current scores for the nodes in the inspect list.
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
    page_names = page_names.zipWithIndex().map(lambda (n, id): (id + 1, n))
    page_names = page_names.sortByKey().cache()

    #######################################################################
    # set up partitioning - we have roughly 16 workers, if we're on AWS with 4
    # nodes not counting the driver.  This is 16 partitions per worker.
    #
    # Cache this result, so we don't recompute the link_string_to_KV() each time.
    #######################################################################
    neighbor_graph = neighbor_graph.partitionBy(256).cache()

    # find Kevin Bacon
    Kevin_Bacon = page_names.filter(lambda (K, V): V == 'Kevin_Bacon').collect()
    # This should be [(node_id, 'Kevin_Bacon')]
    assert len(Kevin_Bacon) == 1
    Kevin_Bacon = Kevin_Bacon[0][0]  # extract node id

    # find Harvard University
    Harvard_University = page_names.filter(lambda (K, V):
                                           V == 'Harvard_University').collect()
    # This should be [(node_id, 'Harvard_University')]
    assert len(Harvard_University) == 1
    Harvard_University = Harvard_University[0][0]  # extract node id

    pagerank(neighbor_graph,
             page_names,
             iterations=10,
             inspect=[Kevin_Bacon, Harvard_University])


#  aws emr create-cluster --name "Spark cluster" --release-label emr-4.1.0 --applications Name=Spark --ec2-attributes KeyName=SparkKeyPair --instance-type m3.xlarge --instance-count 5 --use-default-roles

# spark-submit --num-executors 4 --executor-cores 4 --executor-memory 8g PageRank.py
