"""Producer/Consumer class in which producer adds to the consumer queue in a
multithreaded fashion, and the consumer will work in the same order as the
items added to the producer queue.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from Queue import Queue
except ModuleNotFoundError:
    from queue import Queue
from threading import Thread
from heapq import heappush, heappop


class OrderedProducerConsumer:
    """Producer/Consumer context manager in which producer adds to the consumer
    queue in a multithreaded fashion, and the consumer will work in the same
    order as the items added to the producer queue.
    """
    def __init__(self, num_threads, produce_function, consume_function):
        """ Initializes OrderedProducerConsumer
        Args:
          num_threads: Total number of producer threads.
          produce_function: Function object with input of item put into
            producer queue and returns object to be used as input to the
            consume_function
          consume_function: Function object with with input of item returned by
           produce_function.
        """
        self.num_threads = num_threads
        self.queue_size = self.num_threads * 2
        self.producer_queue = Queue(self.queue_size)
        self.consumer_queue = Queue(self.queue_size)
        self.consumer_list = []
        self.consumed_count = 0
        self.consume_order = 0
        self.produce_function = produce_function
        self.consume_function = consume_function
        self.started = False

    def __enter__(self):
        """ Enter `with` block."""
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        """ Exit `with` block and join threads."""
        self.join()

    def _produce(self):
        """ Producer worker. Polls producer queue and adds items to consumer
        queue.
        """
        while True:
            item, consume_order = self.producer_queue.get()
            produced_item = self.produce_function(item)
            self.consumer_queue.put((produced_item, consume_order))
            self.producer_queue.task_done()

    def _consume(self):
        """ Consumer worker. Polls consumer queue and consumes items in the
        same order as items added to the producer queue.
        """
        while True:
            produced_item, consume_order = self.consumer_queue.get()
            heappush(self.consumer_list, (consume_order, produced_item))
            while consume_order == self.consumed_count:
                self.consume_function(produced_item)
                heappop(self.consumer_list)
                print('Inserted ' + str(self.consumed_count))
                self.consumed_count += 1
                if self.consumer_list:
                    consume_order, produced_item = self.consumer_list[0]
            self.consumer_queue.task_done()


    def start(self):
        """ Start worker threads"""
        self.started = True
        for _ in range(self.num_threads):
            produce_thread = Thread(target=self._produce)
            produce_thread.daemon = True
            produce_thread.start()
        consume_thread = Thread(target=self._consume)
        consume_thread.daemon = True
        consume_thread.start()

    def join(self):
        """ Join worker threads"""
        self.producer_queue.join()
        self.consumer_queue.join()

    def put(self, item):
        """ Put item into producer queue
        Args:
          item: Item to put into producer queue. Will be used as argument to
            producer_function.
        """
        while len(self.consumer_list) > self.queue_size:
            #DONT INSERT IF inserter_list has grown too big, let
            #let producers catch up
            pass

        self.producer_queue.put((item, self.consume_order))
        self.consume_order += 1
