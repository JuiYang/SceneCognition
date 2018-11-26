APP_STL:=gnustl_static
APP_CPPFLAGS:=-frtti -fexceptions 
APP_CPPFLAGS += -std=c++11
APP_ABI:= armeabi-v7a 
APP_BUILD_SCRIPT:=jni/Android.mk
APP_CFLAGS+=-DDLIB_NO_GUI_SUPPORT=on
#NO_USE
LOCAL_ARM_NEON := true




