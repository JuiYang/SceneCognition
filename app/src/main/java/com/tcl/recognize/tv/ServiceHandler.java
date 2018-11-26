package com.tcl.recognize.tv;

import android.content.Context;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import com.tcl.recognize.tslite.ImageClassifier;
import com.tcl.recognize.tslite.LocalRecognize;
import com.tcl.recognize.util.Constant;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

/**
 * Our Handler used to execute operations on the main thread. This is used to
 * schedule increments of our value.
 * 
 * @author taozhang
 * 
 */
public class ServiceHandler extends Handler {
	private static final String TAG = "RecognizeTask";
	// public static int audioNum = 0;
	public static ServiceHandler mServiceHandler;

//	public int mScreenShotInterval = 5;
	public static final String errString_inStarting = "后台服务正在启动";

//	private final ScheduledExecutorService getThirdAppConfigExecutorService = Executors
//			.newSingleThreadScheduledExecutor();

	private final ScheduledExecutorService recognizeExecutorService = Executors.newSingleThreadScheduledExecutor();
	private final ScheduledExecutorService peroidicExecutorService = Executors.newSingleThreadScheduledExecutor();

	private ScheduledFuture<?> peroidicRecognizeTask;

	private RecognizeService mService;

	public static String lastResult = "errResult";
	private LocalRecognize mLocalRecgnize;
	public ImageClassifier imageClassifier;
	private LocalResize localResize;
	
	private int screenShotInterval = 1;

	ServiceHandler(RecognizeService mRecognizeService) {
		super();
		this.mService = mRecognizeService;
		mServiceHandler = this;

	}

	public interface MsgType {
		int MSG_BASE = 100;
		// start
		int START = MSG_BASE - 1;
		// stop
		int STOP = MSG_BASE - 2;

		// get configuration
		int GET_CONFIG_RESULT = MSG_BASE + 8;
		int GET_CONFIG_ERROR = MSG_BASE + 9;
		// recognize task
		int RECOGNIZE_RESULT = MSG_BASE + 10;
		int RECOGNIZE_RESULT_UNKNOWN = MSG_BASE + 11;
		int RECOGNIZE_ERROR = MSG_BASE + 12;
		// channel switch notification
		int CHANNEL_SWITCH_NOTIFICATION = MSG_BASE + 13;
		int CHANNEL_SWITCH_NOTIFICATION_ERROR = MSG_BASE + 14;
		// top activity change
		int PAUSE = MSG_BASE + 15;
		int RESUME = MSG_BASE + 16;

		int LOCAL_INIT_OK = MSG_BASE + 17;

	}

	@Override
	public void handleMessage(Message msg) {
		switch (msg.what) {
		/*
		 * msg.what=START
		 */
		case MsgType.START:
			Log.d(TAG, "schedule startRecognize");
			GetConfig getConf = new GetConfig(this, mService);
//			screenShotInterval = getConf.getScreenshot_interval_time();

			// add by lilan_n for local recognize to load model library
			if (localInit()) {
				// 加载模型success ,开始截屏识别
				schedulePeriodicRecognizeTask(screenShotInterval);
			}

			break;

		case MsgType.STOP:

			shutdownAndAwaitTermination(recognizeExecutorService);
			shutdownAndAwaitTermination(peroidicExecutorService);

			break;

		/*
		 * msg.what=RECOGNIZE_RESULT|RECOGNIZE_ERROR
		 */
		case MsgType.RECOGNIZE_RESULT:
		case MsgType.RECOGNIZE_ERROR:
			Log.d(TAG, "MsgType.what = ");
			switch (msg.what) {
			case MsgType.RECOGNIZE_RESULT:
				String tmp = lastResult;

				lastResult = (String) msg.obj;

				Log.d(TAG, "RECOGNIZE_RESULT=" + lastResult);

				break;
			case MsgType.RECOGNIZE_ERROR:
				Log.d(TAG, "RECOGNIZE_ERROR");

				String errString = (String) msg.obj;
				Log.d(TAG, "errString = " + errString);
				break;
			}
			if(Constant.DEBUG){
				// 显示界面
				RecognizeService.mRecognizeService.refresh(lastResult);
			}
			
			schedulePeriodicRecognizeTask(screenShotInterval);
			
			break;
		}

	}

	private boolean localInit() {
		// 加载模式库
		Log.d(TAG, "localInit");
		Context mContent = RecognizeService.mRecognizeService;
		try {
			copyFile(mContent);

		} catch (Exception e) {
			// TODO: handle exception
			Log.e(TAG, "initTensorflow error");
			return false;
		}
		try {
			localResize = LocalResize.getInstance(mContent);
			mLocalRecgnize = LocalRecognize.getInstance(mContent);
			if (mLocalRecgnize != null) {
				if (mLocalRecgnize.init(mContent)) {
					return true;
				} else
					return false;
			}
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
			return false;
		}
		
		return false;
	}

	public void schedulePeriodicRecognizeTask(int interval) {
		// 防漏查询
		Log.d(TAG, "防漏查询 schedulePeriodicRecognizeTask");
		stopPeroidicRecognizeTask();

		Log.d(TAG, "发起识别任务：new RecognizeTask");
		peroidicRecognizeTask = recognizeExecutorService.schedule(
				new RecognizeTask("PeriodicRecognizeTask", this, mService), interval, TimeUnit.SECONDS);
	}

	public void stopPeroidicRecognizeTask() {
		if (peroidicRecognizeTask != null) {
			peroidicRecognizeTask.cancel(true);
			peroidicRecognizeTask = null;
		}
	}

//	public void scheduleRecognizeTask(int delay) {
//		Log.d(TAG, "scheduleRecognizeTask 单次查询");
//
//		Log.d(TAG, "RecognizeTask.hasNextScreenshot = " + RecognizeTask.hasNextScreenshot);
//		if (!RecognizeTask.hasNextScreenshot) {
//			// recognizeTask =
//			recognizeExecutorService.schedule(new RecognizeTask("RecognizeTask", httpclient, this, mService), delay,
//					TimeUnit.SECONDS);
//		}
//	}

	public static void shutdownAndAwaitTermination(ExecutorService pool) {
		// Disable new tasks from being submitted
		pool.shutdown();
		try {
			// Wait a while for existing tasks to terminate
			if (!pool.awaitTermination(60, TimeUnit.SECONDS)) {
				pool.shutdownNow(); // Cancel currently executing tasks
				// Wait a while for tasks to respond to being cancelled
				if (!pool.awaitTermination(60, TimeUnit.SECONDS))
					System.err.println("Pool did not terminate");
			}
		} catch (InterruptedException ie) {
			// (Re-)Cancel if current thread also interrupted
			ie.printStackTrace();
			pool.shutdownNow();
			// Preserve interrupt status
			Thread.currentThread().interrupt();
		}
	}
	
	
	private void copyFile(Context context) {
		Log.d(TAG, "copyFile");
		try {
			Log.d(TAG, "copyFile start-model1.tflite");

			String fileName = context.getFilesDir() + "/" + Constant.MODEL_PATH;
			InputStream assetsFile;
			// File file = new File(fileName);
			// if (!file.exists()) {
			// LogManagerUtil.d(TAG, fileName + " file.exists() is false");

			try {
				assetsFile = context.getAssets().open(Constant.MODEL_PATH);
				OutputStream fileOut = new FileOutputStream(fileName);
				byte[] buffer = new byte[1024];
				int length;
				while ((length = assetsFile.read(buffer)) > 0) {
					fileOut.write(buffer, 0, length);
				}

				fileOut.flush();
				fileOut.close();
				assetsFile.close();

				Log.d(TAG, "copyFile end-model1.tflite");
			} catch (IOException e) {
				e.printStackTrace();
			}
			// } else {
			// LogManagerUtil.d(TAG, fileName + " file.exists() is true");
			// }
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	
	public void shutDown(){
		shutdownAndAwaitTermination(recognizeExecutorService);
		shutdownAndAwaitTermination(peroidicExecutorService);
	}

}
