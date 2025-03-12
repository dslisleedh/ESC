import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.backends import cudnn

import threading
import time 


class PowerMeasurer:
    def __init__(self, tick_interval=1):
        self.tick_interval = tick_interval
        self.power_usage = []
        self.stop_event = threading.Event()
        self.power_thread = None
    
    def start(self):
        def get_power_usage():
            time.sleep(1)
            while not self.stop_event.is_set():
                self.power_usage.append(torch.cuda.power_draw(0))
                time.sleep(self.tick_interval)
        self.stop_event.clear()
        self.power_thread = threading.Thread(target=get_power_usage)
        self.power_thread.start()
    
    def stop(self):
        self.stop_event.set()
        if self.power_thread is not None:
            self.power_thread.join()
            self.power_thread = None
    
    def average(self):
        if self.power_usage:
            return np.mean(self.power_usage)/1000
        return None


# @calc_average_power_usage()
def test_direct_metrics(model, input_shape, n_repeat=100, use_float16=False, jit_compile=False):
    cudnn.benchmark = True
    
    print(f'CUDNN Benchmark: {cudnn.benchmark}')
    if use_float16:
        context = torch.cuda.amp.autocast
        print('Using AMP(FP16) for testing ...')
    else:
        context = nullcontext
        print('Using FP32 for testing ...')
    
    x = torch.FloatTensor(*input_shape).uniform_(0., 1.)
    x = x.cuda()
    print(f'Input shape: {x.shape}')
    model = model.cuda()
    model.eval()
    
    if jit_compile:
        with torch.no_grad():
            model = torch.jit.trace(model, x)
    
    measure_power = PowerMeasurer()
    
    with context():
        with torch.inference_mode():
            print('warmup ...')
            for _ in tqdm.tqdm(range(100)):  
                model(x)  # Make sure CUDNN to find proper algorithms, especially for convolutions.
                torch.cuda.synchronize()
    
            print('testing ...')
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            
            measure_power.start()
            
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            timings = np.zeros((n_repeat, 1))
            
            for rep in tqdm.tqdm(range(n_repeat)):
                starter.record()
                model(x)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
                
            measure_power.stop()
    
    avg = np.sum(timings) / n_repeat
    med = np.median(timings)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('------------ Results ------------')
    print(f'Average time: {avg:.5f} ms')
    print(f'Median time: {med:.5f} ms') 
    print(f'Maximum GPU memory Occupancy: {torch.cuda.max_memory_allocated() / 1024**2:.5f} MB')
    print(f'Maximum GPU memory Reserved: {torch.cuda.max_memory_reserved() / 1024**2:.5f} MB')
    print(f'Params: {params / 1000}K')  # For convenience and sanity check.
    print(f'Average power usage: {measure_power.average()} W')
    print('---------------------------------')
