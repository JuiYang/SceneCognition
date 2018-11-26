package com.tcl.recognize.util;

import android.app.Application;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import com.tcl.recognize.tv.RecognizeService;

import java.lang.Thread.UncaughtExceptionHandler;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

public class CaughtException extends Application {
	private static final String TAG = "CaughtException";

	private String errorinfo = "";
	private int appErrorNumber = 0x03020001; // 播放器模块错误码
	private Class<?> mClass;
	private Object mErrorReport;
	// private Object mSendReport;
	private int ErrorCodeReportMethod_logcat = 0;// logcat
	private int ErrorCodeReportMethod_file = 1;// file
	private String pkgName;

	@Override
	public void onCreate() {
		// TODO Auto-generated method stub
		super.onCreate();

		try {
			pkgName = this.getPackageName();
			Log.d(TAG, "application onCreate,and this app package name is "
					+ pkgName);
		} catch (Exception e) {
			e.printStackTrace();
		}

		Thread.setDefaultUncaughtExceptionHandler(new UncaughtExceptionHandler() { // 用此方法捕获整个进程中的异常
			@Override
			// 在异常时调用终端管理提供的方法上报异常
			public void uncaughtException(Thread thread, Throwable ex) {

				//add by sunxz for Bug:0071958 20160523 below
				 findErrorReportClass();
				 errorinfo = ex.getLocalizedMessage();
				 Log.d("=======","errorinfo=="+ errorinfo);
				 initErrorReport(errorinfo);
				//add by sunxz for Bug:0071958 20160523 above
				 
				Log.d(TAG, "ex = " + ex);

				RecognizeService.mRecognizeService.stopService(new Intent(
						RecognizeService.mRecognizeService,
						RecognizeService.class));
				
				try {
					Thread.sleep(2000);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				android.os.Process.killProcess(android.os.Process.myPid()); // 进程捕获到异常后将自己关闭

			}
		});

	}

	/**
	 * 通过反射方法获取系统是否存在ErrorReport类
	 * */
	public Class<?> findErrorReportClass() {
		try {
			mClass = Class.forName("com.tcl.terminalmanager.ErrorReport");
			Log.d("=======", "find this class");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			e.printStackTrace();
		}
		return mClass;
	}

/**
 * 通过反射方调用sendReport方法上报错误
 * */
		public void initErrorReport(String errorinfo) {

			//add by sunxz for Bug:0071958 20160523 below
			if(mClass != null){
			//add by sunxz for Bug:0071958 20160523 above

			try {
			Method mReport = mClass.getDeclaredMethod("getInstance",
			Context.class); // 获取ErrorReport.getInstance(context)
			if (mReport != null) {
			mErrorReport = mReport.invoke(mClass, CaughtException.this);
			Method mSendReport = mClass.getDeclaredMethod("sendReport",
			int.class, String.class, int.class, String.class); // 获取
			// m.sendReport(errorCode,
			// errInfo,
			// logStyle,
			// FileName)方法
			if (mSendReport != null) {
			mSendReport.invoke(mErrorReport, appErrorNumber, errorinfo,
			ErrorCodeReportMethod_logcat, null);
			}

			}
		} catch (NoSuchMethodException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

  }
}