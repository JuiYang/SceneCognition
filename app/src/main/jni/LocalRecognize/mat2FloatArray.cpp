//
// Created by tclxa on 18-11-6.
//

#include "mat2FloatArray.h"
#include <iostream>
#include <cstring>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <vector>
#include <android/log.h>
#include "Main.h"
#include "com_tcl_recognize_tv_LocalResize.h"


#include "stdafx.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define LOGD_TAG "localRecognizeCpp"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOGD_TAG, __VA_ARGS__)

#define LOGI_TAG "Recognizelogo"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, LOGI_TAG, __VA_ARGS__))

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";
    std::vector<char> charsCode;
    const jchar *chars = env->GetStringChars(jStr, NULL);
    jsize len = env->GetStringLength(jStr);
    jsize i;
    for (i = 0; i < len; i++) {
        int code = (int) chars[i];
        charsCode.push_back(code);
    }
    env->ReleaseStringChars(jStr, chars);
    return std::string(charsCode.begin(), charsCode.end());
}


extern "C"
jfloatArray Java_com_tcl_weilong_mace_MaceClassifier_maceMat2Array(JNIEnv *env, jclass clazz,jint width,
                                                                 jint high, jstring picPath){
    String picFile = jstring2string(env, picPath);
    LOGD("picPath : %s"+picFile);
    Mat src = imread(path); //从路径名中读取图片
    jfloatArray array1 = env->NewFloatArray(width*high*3);

    int i = 0;
    for(int row = 0; row<src.rows; row++){
        for(int col =0 ; col<src.cols; col++){
            for (int channel =0 ; channel <src.channels(); channel++) {

            array1[i++] = (jfloat)src.at<Vec3b>(row, col)[channel];
            //int g = src.at<Vec3b>(row, col)[1];
            //int r = src.at<Vec3b>(row, col)[2];
            }
        }
    }
    LOGD("mat to array over");
    return array1;
}

