package com.tcl.recognize.util;

import android.util.Log;

import org.apache.log4j.Logger;


public class LogManagerUtil {
	private static final String TAG =LogManagerUtil.class.getSimpleName();

	public static boolean isWriteLog = true;
	public static boolean isStopWriteLog = false;

	public LogManagerUtil() {

	}

	public static void setLogManagerUtil(boolean versionType) {
		Log.d(TAG, "setLogManagerUtil->" + versionType);

		if (versionType) {
			isWriteLog = false;
		} else {
			isWriteLog = true;
		}
		Log.d(TAG, "isWriteLog->" + isWriteLog);
	}

	public static void setWriteLogState(boolean state) {
		Log.d(TAG, "setWriteLogState->" + state);

		isStopWriteLog = state;
		Log.d(TAG, "isStopWriteLog->" + isStopWriteLog);
	}

	public static void d(String TAG, String str) {
		if (!isStopWriteLog) {
			if (isWriteLog) {
				Logger log = Logger.getLogger(TAG);

				// write log
				log.debug(str);
			} else {
				Log.d(TAG, str);
			}
		}
	}

	public static void i(String TAG, String str) {
		if (!isStopWriteLog) {
			if (isWriteLog) {
				Logger log = Logger.getLogger(TAG);

				// write log
				log.info(str);
			} else {
				Log.i(TAG, str);
			}
		}
	}

	public static void e(String TAG, String str) {
		if (!isStopWriteLog) {
			if (isWriteLog) {
				Logger log = Logger.getLogger(TAG);

				// write log
				log.error(str);
			} else {
				Log.e(TAG, str);
			}
		}
	}
}
