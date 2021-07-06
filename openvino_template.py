import numpy as np
from openvino.inference_engine import IECore

ie = IECore()                                            # Instantiate IE core object
model='./public/googlenet-v1/FP16/googlenet-v1'
net = ie.read_network(model+'.xml', model+'.bin')        # Read model file
exenet = ie.load_network(network=net, device_name='CPU') # Load the model to device plugin
dummy_in = np.zeros((1,3,224,224), dtype=np.uint8)       # Prepare dummy input data
res = exenet.infer({'data':dummy_in})                    # Infer ('data' is input blob name)
print(res['prob'].shape)                                 # Inference result is in 'res['prob']'