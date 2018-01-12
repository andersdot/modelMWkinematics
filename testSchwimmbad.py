

data = list(range(10000))

def do_the_processing(x):
    # Do something with each value in the data set!
    # For example, here we just square each value
    return x**2


#the long way
values = []
for x in data:
    values.append(do_the_processing(x))

#using the map() function which applies the function (passed as the first argument) to each element in the iterable (second argument)
#Note: python3 map() returns a generator object, so we call list on this to get out the values
values = list(map(do_the_processing, data))


#class-based wrapper for the built-in (serial) map() function
from schwimmbad import SerialPool
pool = SearlPool()
values = list(pool.map(do_the_processing, data))

#utilize multiple cores on the same processor
from schwimmbad import MultiPool
with MultiPool() as pool:
    values = list(pool.map(do_the_processing, data))

#when using MPI pool tell all worker processes to wait for tasks from the master process
def main(pool, data):
    values = pool.map(do_the_processing, data)



from schwimmbad import MPIPool
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)

#selecting a pool with command-line arguments
#worker function- takes a single 'task' (eg one datum or one object's data) and returns a result based on that task
import math
def worker(task):
    a, b = task
    return math.cos(a) + math.sin(b)

#main function - accepts a pool object and performs the actual processing
def main(pool):
    #generate some fake data
    a = [random.uniform(0, 2*math.pi) for _ in range(10000)]
    b = [random.uniform(0, 2*math.pi) for _ in range(10000)]
    tasks = list(zip(a, b))
    results = pool.map(worker, tasks)
    pool.close()

if __name__ == '__main__':
    import schwimmbad
    from argparse import ArgumentParser
    parser = ArgumentParser(description = 'Schwimmbad example.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--ncores', dest='n_cores', default=1, type=int, help='number of processes (uses multiprocessing).')
    group.add_argument('--mpi', dest='mpi', default=False, action='store_true', help='run with mpi.')
    args = parser.parse_args()
    pool = schwimmbad.choose_pool(mpi=args.mpi, processes=args.ncores)
    main(pool)


import schwimmbad

class Worker(object):
    def __init__(self, output_path):
        self.output_path = output_path
    def work(self, a, b):
        return a*b - b**2
    def callback(self, result):
        with open(self.output_path, 'a') as f:
            f.write('{0}\n'.format(result))
    def __call__(self, task):
        a, b = task
        return self.work(a, b)

def main(pool):
    worker = Worker('output_file.txt')
    tasks = list(zip(range(16384), range(16384)[::-1]))
    for r in pool.map(worker, tasks, callback=worker.callback):
        pass
    pool.close()
