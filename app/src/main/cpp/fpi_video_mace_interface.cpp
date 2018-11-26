// Copyright 2018 Xiaomi, Inc.  All rights reserved.

#include "fpi_video_mace_interface.h"
#include "mace-define.h"
#include <vector>
#include <fstream>
#include <algorithm>
#include <functional>
#include <fstream>
#include <string>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <sstream>
#include <cstring>

string labelContents[] = {"Fallen leaves, yellow leaves",
                        "yellow sand, sand dunes, camel(s)",
                        "forest, green tree(s)",
                        "grass, green",
                        "stone, green trees",
                        "lake, ocean, sea",
                        "others",
                        "face, skin, body",
                        "blue sky, white clouds",
                        "snow, white, snow mountain",
                        "run"};


// 分割字符串
void split(const std::string &s, std::vector<std::string> &elems, char delim = ':') {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
}


int fpi_video_mobile_net_create_engine()
{
	// 设置Attrs
	MaceContext &mace_context = GetMaceContext();
	mace::MaceStatus status;
	// openmp
	status = mace::SetOpenMPThreadPolicy(2,static_cast<mace::CPUAffinityPolicy>(0));

	log2file("image_classify attrs openmp result: status, threads: 2, cpu: 0");
	//  gpu
	mace::SetGPUHints(
			static_cast<mace::GPUPerfHint>(3),
			static_cast<mace::GPUPriorityHint>(3));

	log2file("image_classify attrs gpu perf: 3, priority: 3");
	//  opencl cache
	const char *kernel_path_ptr = kernel_path.c_str();
	log2file("www:convert kernel path");
	if (kernel_path_ptr == nullptr){
		log2file("kernel_path_ptr is nullptr");
		return -1;
	}
	const std::string kernel_file_path(kernel_path_ptr);
	mace_context.storage_factory.reset(
			new mace::FileStorageFactory(kernel_file_path));
	mace::SetKVStorageFactory(mace_context.storage_factory);
	//env->ReleaseStringUTFChars(kernel_path, kernel_path_ptr);
	log2file("www:SetKVStorageFactory error!");
	delete kernel_path_ptr;
	log2file("www:delete kernel_path_ptr error!");


	// 设置engine
	log2file("www:delete model_name_ptr error!");
	//  load model input and output name
	auto model_info_iter =
			mace_context.model_infos.find(mace_context.model_name);
	if (model_info_iter == mace_context.model_infos.end()) {
		log2file("model_info_iter != mace_context.model_infos.end()");
		return -1;
	}
	std::vector<std::string> input_names = { model_info_iter->second.input_name };
	std::vector<std::string> output_names = { model_info_iter->second.output_name };
	mace_context.device_type = ParseDeviceType("CPU");
	log2file("www:delete device_ptr error!");

	mace::MaceStatus create_engine_status =
			CreateMaceEngineFromCode(mace_context.model_name,
									 std::string(),
									 input_names,
									 output_names,
									 mace_context.device_type,
									 &mace_context.engine);
	return create_engine_status == mace::MaceStatus::MACE_SUCCESS ? 0 : -1;
}


int fpi_video_mobile_net_classify(unsigned char * buffer, int length, struct _ST_video_resultlabels *result)
{
	MaceContext& mace_context = GetMaceContext();
	//  prepare input and output
	auto model_info_iter =
			mace_context.model_infos.find(mace_context.model_name);

	if (model_info_iter == mace_context.model_infos.end())
	{
		log2file("Invalid model name");
		return -1;
	}
	const ModelInfo &model_info = model_info_iter->second;
	const std::string &input_name = model_info.input_name;
	const std::string &output_name = model_info.output_name;
	const std::vector<int64_t> &input_shape = model_info.input_shape;
	const std::vector<int64_t> &output_shape = model_info.output_shape;
	const int64_t input_size =
			std::accumulate(input_shape.begin(), input_shape.end(), 1,
							std::multiplies<int64_t>());
	const int64_t output_size =
			std::accumulate(output_shape.begin(), output_shape.end(), 1,
							std::multiplies<int64_t>());
	//  load input
	log2file("www:start load input");

	if (buffer == nullptr)
	{
		log2file("www:input_data_ptr is nullptr");
		return -1;
	}

	if (length != input_size)
	{
		log2file("www:length_intput_data != input_size");
		return -1;
	}

	std::map<std::string, mace::MaceTensor> inputs;
	std::map<std::string, mace::MaceTensor> outputs;
	// construct input
	auto buffer_in = std::shared_ptr<float>(new float[input_size],
											std::default_delete<float[]>());
	std::copy_n(buffer, input_size, buffer_in.get());
	//env->ReleaseFloatArrayElements(input_data, input_data_ptr, 0);
	delete buffer;
	log2file("www:delete input_data_ptr error");
	inputs[input_name] = mace::MaceTensor(input_shape, buffer_in);

	// construct output
	auto buffer_out = std::shared_ptr<float>(new float[output_size],
											 std::default_delete<float[]>());
	outputs[output_name] = mace::MaceTensor(output_shape, buffer_out);

	// run model
	mace_context.engine->Run(inputs, &outputs);

	std::shared_ptr<float> data = outputs[output_name].data();

    // 获取置信度最大值以及对应的下标
    int labelIndex = 0;
    for(int i = 0;i != LABEL_NUM;++i)
    {
        for (int j = 0; j != LABEL_NUM; ++j)
        {
            if (*data+i > *data+j)
            {
                labelIndex = i;

            }else {
                labelIndex = j;
            }
        }
    }

    // 最大置信度
    float degree = *data+labelIndex;
    strcpy(result->contentText, labelContents[labelIndex+1].c_str());
    result->degree = degree;
    result->label = EN_VIDEO_SCENTTYPE(labelIndex);
	// 释放data指针
	delete &data;
	return 0;
}



int fpi_video_mobile_net_release_source()
{
	// 源码中是唯一指针
	MaceContext& mace_context = GetMaceContext();
	// 释放资源
	delete &mace_context;
	return 0;
}

