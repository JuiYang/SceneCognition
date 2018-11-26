package com.tcl.recognize.tv;

import android.content.Context;
import android.os.Handler;
import android.util.Log;

import com.tcl.recognize.util.DeviceInfo;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

public class GetConfig {

	private static final String TAG = "GetConfig";

//	public static GetConfigInfoTask mGetConfigInfoTask;
	public static final String SQA_FILE_PATH = "/data/scr.txt";
	// 默认截屏间隔5s
	private int screenshot_interval_time = 5;

	private Handler mHandler;
	private Context mContext;

	public static int SQA_Screenshot_interval_time;
	public static int DebugMode;

	public GetConfig(Handler mHandler, Context mContext) {
		this.mHandler = mHandler;
		this.mContext = mContext;

		if (DeviceInfo.mDeviceInfo == null) {
			DeviceInfo.getInstance(mContext);
		}		
	}

	public static boolean SQA_MODE = false;
	// public static int Screenshot_Data = 0;

	public void judge_run_mode() {
		// SQA_screenshot_flag = 0 ;
		File file = new File(SQA_FILE_PATH);
		if (file.exists()) {
			SQA_MODE = true;
			getforQATest();
		}
	}

	public void getforQATest() {
		String strLine = null;
		String[] arrLineString = null;
		FileReader localFileReader = null;
		HashMap<String, String> mDataMap = new HashMap<String, String>();
		BufferedReader localBufferedReader = null;
		try {
			localFileReader = new FileReader(SQA_FILE_PATH);
			localBufferedReader = new BufferedReader(localFileReader);
			while (true) {
				strLine = localBufferedReader.readLine();
				if (strLine == null) {
					break;
				}
				arrLineString = strLine.split("=");
				if (arrLineString.length >= 2) {
					mDataMap.put(arrLineString[0], arrLineString[1]);
				}
			}
			localBufferedReader.close();
			localFileReader.close();

			for (String name : mDataMap.keySet()) {
				Log.d(TAG, "name=" + name);

				if ("screenshot_interval_time".equals(name)) {
					String temp = mDataMap.get(name).trim();
					int tp = Integer.parseInt(temp);
					SQA_Screenshot_interval_time = tp;
					Log.d(TAG, "SQA_Screenshot_interval_time=" + SQA_Screenshot_interval_time);
				}
				
				if ("debug_mode".equals(name)) {
					String temp = mDataMap.get(name).trim();
					DebugMode = Integer.parseInt(temp);
					Log.d(TAG, "debug_mode = " + DebugMode);
				}

			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public int getScreenshot_interval_time() {
		/**
		 * SQA mode modify by lilan_n * 20161227
		 */
		judge_run_mode();
		
		if ((SQA_MODE == true) && (Integer.valueOf(SQA_Screenshot_interval_time) != null)) {
			Log.d(TAG, "SQA_Screenshot_interval_time=" + SQA_Screenshot_interval_time);

			screenshot_interval_time = SQA_Screenshot_interval_time;
			// return SQA_setScreenshot_interval_time;
		}
		/**
		 * 范围限制，modify by lilan_n 20161226
		 */
		if ((screenshot_interval_time < 0) || (600 < screenshot_interval_time)) {
			screenshot_interval_time = 5;
		}
		return screenshot_interval_time;
	}
	
	
	
//	public boolean getDebugMode() {
//		/**
//		 * SQA mode modify by lilan_n * 20161227
//		 */
//		judge_run_mode();
//
//		if ((SQA_MODE == true) && (Integer.valueOf(DebugMode) != null)) {
//			Log.d(TAG, "getDebugMode = " + DebugMode);
//
//			if (1 == DebugMode)
//				return true;
//			else
//				return false;
//		}
//		return Constant.DEBUG;
//	}

}
