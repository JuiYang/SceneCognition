
#ifndef MACE_MACE_JNI_H
#define MACE_MACE_JNI_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNICALL Java_com_tcl_weilong_mace_MainActivity_maceMobilenetSetAttrs
        (JNIEnv *, jclass, jint, jint, jint, jint, jstring);


JNIEXPORT jint JNICALL
Java_com_tcl_weilong_mace_MainActivity_maceMobilenetCreateEngine
        (JNIEnv *, jclass, jstring, jstring);


JNIEXPORT jfloatArray JNICALL
Java_com_tcl_weilong_mace_MainActivity_maceMobilenetClassify
        (JNIEnv *, jclass, jfloatArray);

#ifdef __cplusplus
}
#endif
#endif
