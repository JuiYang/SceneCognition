//
// Created by tclxa on 18-11-6.
//

#ifndef SCENECOGNITION_MAT2FLOATARRAY_H
#define SCENECOGNITION_MAT2FLOATARRAY_H

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jfloatArray JNICALL
Java_com_tcl_weilong_mace_MaceClassifier_maceMat2Array
        (JNIEnv *, jclass, jint, jint, jstring);

#ifdef __cplusplus
}
#endif

#endif //SCENECOGNITION_MAT2FLOATARRAY_H
