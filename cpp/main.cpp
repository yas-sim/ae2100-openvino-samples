#include <vector>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

namespace ie = InferenceEngine;

int main(int argc, char *argv[]) {
	ie::Core ie;

	ie::CNNNetwork network;
	network = ie.ReadNetwork("./public/googlenet-v1/FP16/googlenet-v1.xml",
							 "./public/googlenet-v1/FP16/googlenet-v1.bin");
	std::shared_ptr<ie::InputInfo> input_info = network.getInputsInfo().begin()->second;
	std::string input_name = network.getInputsInfo().begin()->first;
	input_info->getPreProcess().setResizeAlgorithm(ie::RESIZE_BILINEAR);
	input_info->setLayout(ie::Layout::NHWC);
	input_info->setPrecision(ie::Precision::U8);
		
	ie::DataPtr output_info = network.getOutputsInfo().begin()->second;
	std::string output_name = network.getOutputsInfo().begin()->first;
	output_info->setPrecision(ie::Precision::FP32);

	ie::ExecutableNetwork executable_network = ie.LoadNetwork(network, "CPU");
	ie::InferRequest infer_request = executable_network.CreateInferRequest();

	cv::Mat image = cv::imread("./car.png");

	ie::TensorDesc tDesc(ie::Precision::U8, 
			{1, 3, static_cast<long unsigned int>(image.rows), static_cast<long unsigned int>(image.cols)}, ie::Layout::NHWC);
	infer_request.SetBlob(input_name, ie::make_shared_blob<uint8_t>(tDesc, image.data));

	infer_request.Infer();

	float* output = infer_request.GetBlob(output_name)->buffer();
	std::cout << "\nresults\n------------------" << std::endl;
	std::vector<int> idx;
	for(int i=0; i<1000; i++) idx.push_back(i);
	std::sort(idx.begin(), idx.end(), 
			 [output](const int& left, const int& right) { return output[left]>output[right]; } );
	for (size_t id = 0; id < 5; ++id) {
		std::cout << id <<  " : " << idx[id] << " : " << output[idx[id]]*100 << "% " << std::endl;
	}
}
