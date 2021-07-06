import time
import threading
from tqdm import tqdm, trange

from openvino.inference_engine import IECore, WaitMode, StatusCode
import numpy as np
import copy

#model='./public/resnet-50/FP16/resnet-50'
model='./public/googlenet-v1/FP16/googlenet-v1'

config = [
    [
        'GPU',
        {
            'TUNING_MODE'             : 'TUNING_DISABLED',
            'CLDNN_PLUGIN_PRIORITY'   : '0',
            'CLDNN_PLUGIN_THROTTLE'   : '0',
            'GPU_THROUGHPUT_STREAMS'  : 'GPU_THROUGHPUT_AUTO',
            'EXCLUSIVE_ASYNC_REQUESTS': 'NO'
        }
    ],
    [
        'CPU',
        {
            'CPU_THREADS_NUM'        : '4',
            'CPU_THROUGHPUT_STREAMS' : '2',
        }
    ],
    [
        'MYRIAD',
        {
            'VPU_HW_STAGES_OPTIMIZATION' : 'YES'
        }
    ],
    [
        'HDDL',
        {
            #'VPU_HDDL_GRAPH_TAG'        : '',
            #'VPU_HDDL_STREAM_ID'        : '',
            #'VPU_HDDL_DEVICE_TAG'       : '',
            #'VPU_HDDL_BIND_DEVICE'      : 'NO',
            #'VPU_HDDL_RUNTIME_PRIORITY' : '0'
        }
    ]
]

device = 'HDDL'
niter = 1000

print('device', device, 'iteration', niter, 'model', model)
ie = IECore()
print('read_network()...', end='', flush=True)
net = ie.read_network(model+'.xml', model+'.bin')
print('done', flush=True)

print('set_config()...', end='', flush=True)
for dev, cfg in config:
    if dev == device:
        ie.set_config(cfg, dev)                     # Set plugin configuration
        print(dev, cfg)

print('load_network()...', end='', flush=True)
exenet = ie.load_network(net, device, num_requests=4)
print('done', flush=True)

inputBlobName  = next(iter(net.inputs))
outputBlobName = next(iter(net.outputs)) 
inputShape  = net.inputs [inputBlobName ].shape
outputShape = net.outputs[outputBlobName].shape

dummy_data = np.zeros(inputShape)

completion_count = 0
inuse = [ threading.Lock() for dmy in exenet.requests ]  # flags to block next inference until the previous result is processed
start_time = time.perf_counter()

def callback(status, pydata):
    global completion_count, inuse, exenet, outputBlobName,time
    request_id = pydata
    res = exenet.requests[request_id].outputs[outputBlobName]

    # TODO: process infer result stored in 'res'
    time.sleep(1/1000)      # dummy wait - post process

    completion_count += 1
    inuse[request_id].release()


for n in trange(niter):
    time.sleep(1/1000)      # dummy wait - preprocess

    request_id = exenet.get_idle_request_id()
    if request_id == -1:
        exenet.wait(num_requests=1, timeout=WaitMode.RESULT_READY)
        request_id = exenet.get_idle_request_id()
    inuse[request_id].acquire()

    infreq = exenet.requests[request_id]
    infreq.set_completion_callback(callback, request_id)

    infreq.async_infer(inputs={ inputBlobName : dummy_data } )

while completion_count < niter:
    pass

end_time = time.perf_counter()

print('{:6.2f} sec'.format(end_time - start_time))
print('{:6.2f} inf/sec'.format(niter/(end_time-start_time)))
