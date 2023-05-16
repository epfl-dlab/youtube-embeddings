import numpy as np
import threading
import time
import functools
import innertube
import logging
import multiprocessing as mp
import signal
import traceback

from functools import wraps
from multiprocessing.managers import BaseManager
from multiprocessing import Manager, Lock, Pool
from datetime import timedelta, datetime
from typing import List



class TimedExecException(Exception):
    pass


class MaxRunException(Exception):
    pass

def run_func_timer(t, func, *args, **kwargs):
    
    # Register an handler for the timeout
    def handler(signum, frame):
        raise MaxRunException('Running for too long my boy')
    
    # Register the signal function handler
    signal.signal(signal.SIGALRM, handler)
    
    # Define a timeout for your function
    signal.alarm(t)

    try:
        retval = func(*args, **kwargs)
        signal.alarm(0)
    except MaxRunException:
        return MaxRunException()
    
    return retval

# custom manager to support custom classes
class CustomManager(BaseManager):
    pass

class ProxyClient:
    
    def __init__(self, wrap_client, available=True, wait_until=None, cooldown=timedelta(seconds=5*60)):
        
        self.wrap_client = wrap_client
        self.available = available
        self.cooldown = cooldown
        
        if wait_until is None:
            self.wait_until = datetime(1,1,1)
        else:
            self.wait_until = wait_until

    def __repr__(self):
        return f'Proxy(wrap_client={self.wrap_client}, available={self.available}, wait_until={self.wait_until})'
    
class ClientPool:
    
    def __init__(self, proxies: List[ProxyClient]):
        
        # monkey patching an index on them
        for i,proxy in enumerate(proxies):
            proxy._index = i
            
        self.proxies = proxies
        self.lock = Lock()
        
    def request_proxy(self, current_proxy=None):
        tmp = None
        
        with self.lock:
            
            for proxy in self.proxies:
                if proxy.available and datetime.now() > proxy.wait_until:
                    
                    proxy.available = False
                    tmp = proxy
                    
                    if current_proxy is not None:
                        current_index = current_proxy._index
                        badproxy = self.proxies[current_index]
                        badproxy.wait_until = datetime.now() + badproxy.cooldown
                        badproxy.available = True
                    
                    break
                    
        return tmp
    
    def debug_print(self):
        logging.info(self.proxies)
        
    def get_status(self):
        avail = 0
        broken = 0
        
        for proxy in self.proxies:
            if proxy.available:
                avail += 1
            if proxy.wait_until > datetime.now():
                broken += 1
                
        return avail, broken
            
            
    def proxies_available(self):
        for i,proxy in enumerate(self.proxies):
            if proxy.available and datetime.now() > wait_until:
                return True
    
    def release_proxy(self, proxy):
        with self.lock:
            proxy_release = self.proxies[proxy._index]
            
            proxy_release.available = True
            

def compute_with_pool(pool, iterable, computation_with_client, time_out, postprocess,function_kwargs=None, delay=1):
    
    if function_kwargs is None:
        function_kwargs = {}
    
    # get access to client
    proxy = None
    while proxy is None:
        proxy = pool.request_proxy()
        time.sleep(1)
        
    client = innertube.InnerTube("WEB", proxies=proxy.wrap_client)
    
    def compute():
        all_results = []
        
        for item in iterable:
            
            partial_result = computation_with_client(client, item, **function_kwargs)
            
            all_results.append(partial_result)
            
        time.sleep(delay)
        
        return all_results
        
    ret = run_func_timer(time_out, compute)
    
    if isinstance(ret, MaxRunException):
        logging.info(f'Timeout on {proxy.wrap_client}')
        pool.release_proxy(proxy)
        return TimedExecException(iterable)
        
    #logging.info(f'Finished with client {proxy.wrap_client}')
    pool.release_proxy(proxy)
            
    return postprocess(iterable, ret)

def retry(tries=3, delay=1):
    def take_func(func):
        @wraps(func)
        def trying(*args, **kwargs):
            n_tries = tries
            while n_tries > 0:
                try:
                    return func(*args, **kwargs)
                except MaxRunException:
                    raise
                except Exception as e:
                    logging.warning(f"Error while running {func.__name__} on {args} : {traceback.format_exc()}")
                    time.sleep(delay)
                finally:
                    n_tries -= 1
                
        return trying
    return take_func


def compute_with_proxies(proxies, iterable, iterfunc, postfunc, time_out, chunksize=20, njobs=5, delay=1):
    
    results = []
    
    # create splits
    num_splits = int(np.ceil(len(iterable)/chunksize))
    all_splits = np.array_split(iterable, num_splits)

    # received splits
    rec_splits = 0
    
    CustomManager.register('ClientPool', ClientPool)
    
    with CustomManager() as manager:

        shared_pool = manager.ClientPool([ProxyClient(proxy) for proxy in proxies])
        
        def print_debug():
            t = threading.current_thread()
            while getattr(t, "do_run", True):
                avail, broken = shared_pool.get_status()
                print(f'Available : {avail}, TO: {broken}, progress: {rec_splits}/{num_splits:<10d}', end='\r')
                time.sleep(1)
        

        screen_printing_thread = threading.Thread(
            target=print_debug
        )
        screen_printing_thread.start()

        with Pool(njobs, maxtasksperchild=1) as p:
            
            new_tasks = all_splits
            
            while new_tasks:
                a=new_tasks
                new_tasks=[]
                for result in p.imap_unordered(functools.partial(compute_with_pool,
                                                                 shared_pool,
                                                                 computation_with_client=iterfunc,
                                                                 time_out=time_out,
                                                                 postprocess=postfunc,
                                                                 delay=delay
                                                                ), a):
                    
                    if isinstance(result,TimedExecException):
                        logging.info('TimeoutError, retry')
                        new_tasks.append(result.args[0])
                    else:
                        rec_splits += 1
                        results.append(result)
                    
    
    screen_printing_thread.do_run = False
    screen_printing_thread.join()
                        
    return results



    