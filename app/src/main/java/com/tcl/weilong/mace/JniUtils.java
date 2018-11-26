package com.tcl.weilong.mace;

public class JniUtils {
    static {
        System.loadLibrary("mace_jni");
    }

    /**
     * 给模型设置参数
     * @param ompNumThreads
     * @param cpuAffinityPolicy
     * @param gpuPerfHint
     * @param gpuPriorityHint
     * @param kernelPath
     * @return
     */

    public static  native int maceMobilenetSetAttrs(int ompNumThreads, int cpuAffinityPolicy, int gpuPerfHint, int gpuPriorityHint, String kernelPath);
    /**
     * 给模型创建运行环境
     * @param model
     * @param device
     * @return
     */
    public static native int maceMobilenetCreateEngine(String model, String device);

    /**
     * 模型具体核心功能，识别图片
     * @param input
     * @return
     */
    public static native float[] maceMobilenetClassify(float[] input);


    // mace test
//    public native float[] maceMobileTest(int omp_num_threads, int cpu_affinity_policy,
//                                                int gpu_perf_hint, int gpu_priority_hint, String kernal_name,
//                                                String model_name_str, String device, String file_name);

    public static native float[] maceMobileTest(float[] file_name);
}
