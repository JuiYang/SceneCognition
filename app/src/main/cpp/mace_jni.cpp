#include "src/main/cpp/mace_jni.h"
#include <jni.h>
#include <algorithm>
#include <functional>
#include <map>
#include <tr1/memory>
#include <string>
#include <vector>
#include <numeric>

#include "public/mace.h"
#include "public/mace_runtime.h"
#include "public/mace_engine_factory.h"
#include <android/log.h>
#define  LOG_TAG    "wwww"
#define LOGD(...)  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace {

    struct ModelInfo {
        std::string input_name;
        std::string output_name;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
    };
    struct MaceContext {
        std::shared_ptr<mace::MaceEngine> engine;
        std::shared_ptr<mace::KVStorageFactory> storage_factory;
        std::string model_name;
        mace::DeviceType device_type = mace::DeviceType::CPU;
        std::map<std::string, ModelInfo> model_infos = {
                {"model1", {"input", "MobilenetV2/Predictions/Reshape_1",
                                   {1, 224, 224, 3}, {1, 1001}}},
                {"model2", {"inputs", "outputsss",
                                   {1, 256, 256, 3}, {1, 10}}},
                {"mobilenet_v2", {"inputs", "outputsss",
                                   {1, 128, 128, 3}, {1, 11}}}
        };
    };

    mace::DeviceType ParseDeviceType(const std::string &device) {
        if (device.compare("CPU") == 0) {
            return mace::DeviceType::CPU;
        } else if (device.compare("GPU") == 0) {
            return mace::DeviceType::GPU;
        } else if (device.compare("HEXAGON") == 0) {
            return mace::DeviceType::HEXAGON;
        } else {
            return mace::DeviceType::CPU;   //默认是返回CPU
        }
    }

    MaceContext& GetMaceContext() {
        static auto *mace_context = new MaceContext;

        return *mace_context;
    }

}   //namcspace
/**
 * java+包名+类名+函数名
 * Java+com.tcl.weilong.mace+JniUtils+maceMobilenetSetAttrs
 */
extern "C"
jint Java_com_tcl_weilong_mace_JniUtils_maceMobilenetSetAttrs(JNIEnv *env, jobject instance,
                                                              jint ompNumThreads, jint cpuAffinityPolicy,
                                                              jint gpuPerfHint, jint gpuPriorityHint,
                                                              jstring kernelPath_) {
    MaceContext &mace_context = GetMaceContext();
    mace::MaceStatus status;
    // openmp ？？
    status = mace::SetOpenMPThreadPolicy(
            ompNumThreads,
            static_cast<mace::CPUAffinityPolicy>(cpuAffinityPolicy));
    //  gpu
    mace::SetGPUHints(
            static_cast<mace::GPUPerfHint>(gpuPerfHint),
            static_cast<mace::GPUPriorityHint>(gpuPriorityHint));
    //  opencl cache
    const char *kernel_path_ptr = env->GetStringUTFChars(kernelPath_, nullptr);
    if (kernel_path_ptr == nullptr) return JNI_ERR;
    LOGD("maceMobilenetSetAttrs: kernel_path_ptr is not null");
    const std::string kernel_file_path(kernel_path_ptr);
    mace_context.storage_factory.reset(
            new mace::FileStorageFactory(kernel_file_path));
    mace::SetKVStorageFactory(mace_context.storage_factory);
    env->ReleaseStringUTFChars(kernelPath_, kernel_path_ptr);
    return JNI_OK;
}
extern "C"
jint Java_com_tcl_weilong_mace_JniUtils_maceMobilenetCreateEngine(JNIEnv *env, jobject instance,
                                                                  jstring model_, jstring device_) {
    MaceContext &mace_context = GetMaceContext();
    //  parse model name
    const char *model_name_ptr = env->GetStringUTFChars(model_, nullptr);
    if (model_name_ptr == nullptr) return JNI_ERR;
    LOGD("maceMobilenetCreateEngine: model_name_ptr is not null");
    mace_context.model_name.assign(model_name_ptr);
    env->ReleaseStringUTFChars(model_, model_name_ptr);

    //  load model input and output name
//    auto model_info_iter = mace_context.model_infos.find(mace_context.model_name);
    auto model_info_iter = mace_context.model_infos.find(mace_context.model_name);
    if (model_info_iter == mace_context.model_infos.end()) {
        return JNI_ERR;
    }
    LOGD("model_info_iter == mace_context.model_infos.end()");
    std::vector<std::string> input_names = {model_info_iter->second.input_name};

    std::vector<std::string> output_names = {model_info_iter->second.output_name};

    // get device
    const char *device_ptr = env->GetStringUTFChars(device_, nullptr);
    if (device_ptr == nullptr) return JNI_ERR;
    LOGD("device_ptr is not null");
    mace_context.device_type = ParseDeviceType(device_ptr);
    env->ReleaseStringUTFChars(device_, device_ptr);

    mace::MaceStatus create_engine_status =
            CreateMaceEngineFromCode(mace_context.model_name,
                                     std::string(),
                                     input_names,
                                     output_names,
                                     mace_context.device_type,
                                     &mace_context.engine);


    return create_engine_status == mace::MaceStatus::MACE_SUCCESS ?
           JNI_OK : JNI_ERR;
}extern "C"
jfloatArray Java_com_tcl_weilong_mace_JniUtils_maceMobilenetClassify(JNIEnv *env, jobject instance,
                                                                     jfloatArray input_) {
    MaceContext &mace_context = GetMaceContext();
    //  prepare input and output
    auto model_info_iter =
            mace_context.model_infos.find(mace_context.model_name);
    if (model_info_iter == mace_context.model_infos.end()) {
        return nullptr;
    }
    LOGD("maceMObilenetClassify: model_info_iter == mace_context.model_infos.end()");
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
    jfloat *input_data_ptr = env->GetFloatArrayElements(input_, nullptr);
    if (input_data_ptr == nullptr) return nullptr;
    LOGD("input_data_ptr is not null");
    jsize length = env->GetArrayLength(input_);
    if (length != input_size) return nullptr;
    LOGD("length: %l, input_size: %l", length, input_size);

    std::map<std::string, mace::MaceTensor> inputs;
    std::map<std::string, mace::MaceTensor> outputs;
    // construct input
    auto buffer_in = std::shared_ptr<float>(new float[input_size],
                                            std::default_delete<float[]>());
    std::copy_n(input_data_ptr, input_size, buffer_in.get());
    env->ReleaseFloatArrayElements(input_, input_data_ptr, 0);
    inputs[input_name] = mace::MaceTensor(input_shape, buffer_in);

    // construct output
    auto buffer_out = std::shared_ptr<float>(new float[output_size],
                                             std::default_delete<float[]>());
    outputs[output_name] = mace::MaceTensor(output_shape, buffer_out);

    LOGD("-------------------------------------------");
    mace_context.engine->Run(inputs, &outputs);
    LOGD("++++++++++++++++++++++++++++++++++++++++++++");

    jfloatArray jOutputData = env->NewFloatArray(output_size);  // allocate
    if (jOutputData == nullptr) return nullptr;
    LOGD("jOutputData is not null");
    env->SetFloatArrayRegion(jOutputData, 0, output_size,
                             outputs[output_name].data().get());  // copy
    return jOutputData;
}