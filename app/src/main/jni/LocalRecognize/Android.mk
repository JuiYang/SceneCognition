LOCAL_PATH:=$(call my-dir)

include $(CLEAR_VARS)
PROJECT_CV_ROOT:=$(LOCAL_PATH)/OpenCVlibs
OPENCV_CAMERA_MODULES:=on
OPENCV_LIB_TYPE:=SHARED
OPENCV_INSTALL_MODULES:=on
include $(PROJECT_CV_ROOT)/sdk/native/jni/OpenCV.mk

LOCAL_C_INCLUDES+=$(LOCAL_PATH)
LOCAL_MODULE := mat2Array
LOCAL_SRC_FILES := mat2FloatArray.cpp
LOCAL_LDLIBS:=-llog -ldl -landroid 

#LOCAL_STATIC_LIBRARIES += android_native_app_glue

include $(BUILD_SHARED_LIBRARY)
