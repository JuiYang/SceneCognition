/* Header for class com_xiaomi_mace_JniMaceUtils */


#ifndef MACE_EXAMPLES_ANDROID_MACELIBRARY_SRC_MAIN__DEFINE_H_
#define MACE_EXAMPLES_ANDROID_MACELIBRARY_SRC_MAIN__DEFINE_H_


#include "mace.h"
#include "mace_runtime.h"
#include "mace_engine_factory.h"

#define MACE_LOG_FILE "/mnt/sdcard/SceneRecognition/log.txt"
#define MACE_LABELS_FILE "/storage/sdcard0/0/labels.txt"
#define LABEL_NUM 11

#ifdef __cplusplus
extern "C" {
#endif


namespace {

    const std::string &kernel_path = "/storage/sdcard0/0/mace";

    struct ModelInfo {
        std::string input_name;
        std::string output_name;
        std::vector<int64_t> input_shape;
        std::vector<int64_t> output_shape;
    };

    struct MaceContext {
        std::shared_ptr<mace::MaceEngine> engine;
        std::shared_ptr<mace::KVStorageFactory> storage_factory;
        std::string model_name = "mobilenet_v2";
        mace::DeviceType device_type = mace::DeviceType::CPU;
        std::map<std::string, ModelInfo> model_infos = {
                {"model1", {"input", "MobilenetV2/Predictions/Reshape_1",
                                   {1, 224, 224, 3}, {1, 1001}}},
                {"model2", {"inputs", "outputsss",
                                   {1, 256, 256, 3}, {1, 11}}},
                {"mobilenet_v2", {"inputs", "outputsss",
                                   {1, 128, 128, 3}, {1, 11}}}
        };
    };

    mace::DeviceType ParseDeviceType(const std::string &device) {
        if (device.compare("CPU") == 0) {
            return mace::DeviceType::CPU;
        }
        else if (device.compare("GPU") == 0) {
            return mace::DeviceType::GPU;
        }
        else if (device.compare("HEXAGON") == 0) {
            return mace::DeviceType::HEXAGON;
        }
        else {
            return mace::DeviceType::CPU;
        }
    }

    MaceContext & GetMaceContext() {
        // stay for the app's life time, only initialize once
        static auto *mace_context = new MaceContext;
        return *mace_context;
    }

    static void log2file(const char* msg)
    {
        FILE* fp = fopen(MACE_LOG_FILE, "a");
        if (!fp) return;
        fprintf(fp, "%s\n", msg);
        fclose(fp);
    }

}   //namespace

#ifdef __cplusplus
}
#endif
#endif
