from tqdm import tqdm, trange
from openvino.inference_engine import IECore, WaitMode, StatusCode
import numpy as np

model='./public/googlenet-v1/FP16/googlenet-v1'

device = 'HDDL'
niter = 1000

ie = IECore()
net = ie.read_network(model+'.xml', model+'.bin')
exenet = ie.load_network(net, device, num_requests=4)

dummy_data = np.zeros((1,3,224,224))

completion_count = 0

def callback(status, pydata):
    global completion_count, exenet
    req_id = pydata
    res = exenet.requests[req_id].outputs['prob']
    # TODO: Do inference postprocess

    completion_count += 1


for n in trange(niter):
    request_id = exenet.get_idle_request_id()
    if request_id == -1:
        exenet.wait(num_requests=1, timeout=WaitMode.RESULT_READY)
        request_id = exenet.get_idle_request_id()
    infreq = exenet.requests[request_id]
    infreq.set_completion_callback(callback, request_id)
    infreq.async_infer(inputs={ 'data' : dummy_data } )

while completion_count < niter:
    pass
