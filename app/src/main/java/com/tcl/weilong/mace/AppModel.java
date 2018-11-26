package com.tcl.weilong.mace;


import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;


import java.io.File;
import java.nio.FloatBuffer;

public class AppModel {
    private String model;
    private String device = "";
    private int ompNumThreads;
    private int cpuAffinityPolicy;
    private int gpuPerfHint;
    private int gpuPriorityHint;
    private String kernelPath = "";
    private Handler mJniThread;
    public static final String[] MODELS = new String[]{"mdoel1", "model2", "mobilenet_v2"};
    public static final String[] DEVICES = new String[]{"CPU", "GPU"};
    public AppModel() {
        Log.i("wwww", "AppModel: constructure ...");
        HandlerThread thread = new HandlerThread("jniThread");
        thread.start();
        mJniThread = new Handler(thread.getLooper());
//        model = MODELS[1];
        model = MODELS[2];
        device = DEVICES[0];
        ompNumThreads = 2;
        cpuAffinityPolicy = 0;
        gpuPerfHint = 3;
        gpuPriorityHint = 3;
        kernelPath = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "mace";  //内核路径
        File file = new File(kernelPath);
        if (!file.exists()) {
            file.mkdir();
        }
        Log.i("wwww", "AppModel kernelPath: "+ kernelPath);
    }
    /**
     * 给模型设置属性
     */
    public void maceMobilenetSetAttrs(){
        Log.i("wwww", "maceMobilenetSetAttrs: ");
        int attrs;
        attrs = JniUtils.maceMobilenetSetAttrs(ompNumThreads, cpuAffinityPolicy, gpuPerfHint, gpuPriorityHint, kernelPath);
    }

    /**
     * 给模型创建运行引擎，cpu或者gpu
     */
    public void maceMobilenetCreateEngine(){
        Log.i("wwww", "maceMobilenetCreateEngine: ");
        int engine;
        engine = JniUtils.maceMobilenetCreateEngine(model, device);
    }

    /**
     * 模型的分类函数
     * @param input 缓存的像素点
     * @return  识别结果数据，在类标中匹配识别
     */
    public float[] maceMobilenetClassify(FloatBuffer input){
        Log.i("wwww", "maceMobilenetClassify: ");
        final float[] result;
        result = JniUtils.maceMobilenetClassify(input.array());
        return result;
    }

    public float[] maceMobileTest(FloatBuffer input){
        Log.i("wwww", "maceMobileTest: ");
        float[] result = JniUtils.maceMobileTest(input.array());
        return result;
    }
}
