import time
from tqdm import tqdm, trange

from openvino.inference_engine import IECore, WaitMode, StatusCode
import numpy as np

#model='./public/resnet-50/FP16/resnet-50'
model='./public/googlenet-v1/FP16/googlenet-v1'

device = 'HDDL'
niter = 1000

print('device', device, 'iteration', niter, 'model', model)
ie = IECore()
print('read_network()...', end='', flush=True)
net = ie.read_network(model+'.xml', model+'.bin')
print('done', flush=True)
print('load_network()...', end='', flush=True)
exenet = ie.load_network(net, device)
print('done', flush=True)
inputBlobName  = next(iter(net.inputs))
outputBlobName = next(iter(net.outputs)) 
inputShape  = net.inputs[inputBlobName ].shape
outputShape = net.outputs   [outputBlobName].shape

dummy_data = np.zeros(inputShape)


start_time = time.perf_counter()

for cnt in tqdm(range(niter)):
    time.sleep(1/1000)      # dummy wait - pre-process

    res = exenet.infer(inputs={inputBlobName:dummy_data})

    # TODO: process infer result stored in 'res'
    time.sleep(1/1000)      # dummy wait - post-process

end_time = time.perf_counter()

print('{:6.2f} sec'.format(end_time - start_time))
print('{:6.2f} inf/sec'.format(niter/(end_time-start_time)))
