from tqdm import tqdm, trange

from openvino.inference_engine import IECore, WaitMode, StatusCode
import numpy as np

model='./public/googlenet-v1/FP16/googlenet-v1'

device = 'HDDL'
niter = 1000

ie = IECore()
net = ie.read_network(model+'.xml', model+'.bin')
exenet = ie.load_network(net, device, num_requests=16)

dummy_data = np.zeros((1,3,224,224))

completion_count = 0
inuse = [ False for dmy in exenet.requests ]  # flags to block next inference until the previous result is processed

bar=tqdm(total=niter)
while completion_count<niter:
    
    request_id = exenet.get_idle_request_id()
    if request_id == -1:
        exenet.wait(num_requests=1, timeout=WaitMode.RESULT_READY)
        request_id = exenet.get_idle_request_id()
    infreq = exenet.requests[request_id]
    if inuse[request_id] == True:
        res = infreq.outputs['prob']
        # TODO: process infer result stored in 'res'
        completion_count += 1
        bar.update(1)
    inuse[request_id] = True

    infreq.async_infer(inputs={ 'data' : dummy_data } )
