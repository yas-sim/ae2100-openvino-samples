from tqdm import tqdm, trange

from openvino.inference_engine import IECore
import numpy as np

model='./public/googlenet-v1/FP16/googlenet-v1'

device = 'HDDL'
niter = 1000

ie = IECore()
net = ie.read_network(model+'.xml', model+'.bin')
exenet = ie.load_network(net, device)

dummy_data = np.zeros((1,3,224,224))

for cnt in tqdm(range(niter)):
    res = exenet.infer(inputs={'data' : dummy_data})
    # TODO: process infer result stored in 'res'
