package com.tcl.recognize.tv;


import android.content.Context;
import android.util.Log;

public class LocalResize {
	private static String TAG = LocalResize.class.getSimpleName();

	static {
		Log.d(TAG, "System.loadLibrary->opencv");
		System.loadLibrary("opencv_java3");
		System.loadLibrary("localResize");
	}
	
	public static LocalResize mLocalRecognize;
	public static LocalResize getInstance(Context context) {
		Log.d(TAG, "getInstance->" + mLocalRecognize);

		if (mLocalRecognize == null) {
			mLocalRecognize = new LocalResize();
		}
		return mLocalRecognize;
	}
	
	

	/**
	 * 
	 * @param modelPath 模型路径
	 * @param minHandWidth 可检测最小范围
	 * @param maxHandWidth 可检测最大范围
	 * @return init()执行无问题则返回0，出错则返回负值
	 */
	public native int Init(String modelPath, int minHandWidth, int maxHandWidth);

	/**
	 * 
	 * @param imgPath 被检测图片路径
	 * @return 函数执行出错时返回负值，函数执行正确时返回检测到的手掌的个数(>=0)
	 */
	public native int Detect(String imgPath, String maskPath, String resultPath, int thresh);

//	public native HandRecInfo[] HandRec(int size);
	
	public native int Resize(String imgPath, int size, String resultPath);

}
