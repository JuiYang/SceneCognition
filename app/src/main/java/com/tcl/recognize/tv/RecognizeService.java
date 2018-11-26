package com.tcl.recognize.tv;

import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.graphics.PixelFormat;
import android.os.IBinder;
import android.util.Log;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.TextView;

import com.tcl.recognize.mace.R;
import com.tcl.recognize.tv.ServiceHandler.MsgType;
import com.tcl.recognize.util.Constant;
import com.tcl.recognize.util.DeviceInfo;

import org.apache.log4j.Level;

import java.io.File;

import de.mindpipe.android.logging.log4j.LogConfigurator;

public class RecognizeService extends Service {

	public static final String TAG = "RecognizeService";
	public static RecognizeService mRecognizeService;
	private ServiceHandler mHandler = null;
	public int versionCode;
	public String versionName;

	@Override
	public void onCreate() {
		super.onCreate();

		// ningyb config log4j for write log
		LogConfigurator logConfigurator = new LogConfigurator();
		File saveFile = new File(getFilesDir() + "/dailyLog");
		if (saveFile.isDirectory() == false) {
			saveFile.mkdirs();// 创建下载目录
		}
		logConfigurator.setFileName(getFilesDir() + "/dailyLog/dailyLog.log");
		logConfigurator.setRootLevel(Level.DEBUG);
		logConfigurator.setLevel("org.apache", Level.ERROR);
		logConfigurator.setFilePattern("%d %-5p [%c{3}]-[%L] %m%n");
		logConfigurator.setMaxFileSize(1024 * 1024 * 1);// 文件最大1M
		logConfigurator.setImmediateFlush(true);
		logConfigurator.configure();

		mRecognizeService = this;
		mHandler = new ServiceHandler(this);
		// print version information
		try {
			PackageManager pm = getPackageManager();
			PackageInfo packInfo = pm.getPackageInfo(getPackageName(), 0);
			versionCode = packInfo.versionCode;
			versionName = packInfo.versionName;
			Log.d(TAG, "versionCode=" + PaddingVersionCodeto4Char(versionCode) + ";versionName=" + versionName);
		} catch (NameNotFoundException e) {
			e.printStackTrace();
		}

		mHandler.obtainMessage(MsgType.START).sendToTarget();
		DeviceInfo.mDeviceInfo = DeviceInfo.getInstance(this);
		if (Constant.DEBUG) {
			init_Bubble();
			wm.addView(vbubble, params);
		}

		cleardailyLog();
	}

	// 显示：
	public WindowManager wm;
	private WindowManager.LayoutParams params;
	private View vbubble;
	public static TextView recognizeResult;
	// private static TextView cloudReply;
	// private static TextView cloudTips;

	private void init_Bubble() {
		Log.d(TAG, "   init_Bubble = ");
		wm = (WindowManager) this.getSystemService(Context.WINDOW_SERVICE);// 获取windowManager
		params = new WindowManager.LayoutParams();

		// 设置window type
		params.type = WindowManager.LayoutParams.TYPE_SYSTEM_ALERT;
		/*
		 * 如果设置为params.type = WindowManager.LayoutParams.TYPE_PHONE; 那么优先级会降低一些,
		 * 即拉下通知栏不可见
		 */

		params.format = PixelFormat.RGBA_8888; // 设置图片格式，效果为背景透明
		vbubble = LayoutInflater.from(this).inflate(R.layout.activity_main, null);

		// 设置Window flag

		params.flags = WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL | WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE;
		/*
		 * 下面的flags属性的效果形同“锁定”。 悬浮窗不可触摸，不接受任何事件,同时不影响后面的事件响应。
		 * wmParams.flags=LayoutParams.FLAG_NOT_TOUCH_MODAL |
		 * LayoutParams.FLAG_NOT_FOCUSABLE | LayoutParams.FLAG_NOT_TOUCHABLE;
		 */

		params.width = WindowManager.LayoutParams.WRAP_CONTENT;
		params.height = WindowManager.LayoutParams.WRAP_CONTENT;

		params.format = PixelFormat.RGBA_8888; // 图片格式全透明,去掉黑边
		params.gravity = (Gravity.BOTTOM | Gravity.LEFT);

		recognizeResult = (TextView) vbubble.findViewById(R.id.recognizeResult);
		recognizeResult.setText(this.getString(R.string.title2) + "\n");
		// cloudReply = (TextView) vbubble.findViewById(R.id.cloudReply);
		// cloudReply.setText("");
		// cloudTips = (TextView) vbubble.findViewById(R.id.cloudTips);
		// cloudTips.setText("");

	}

	public void refresh(String lastResult) {
		recognizeResult.setText(lastResult + "\n");
		wm.updateViewLayout(vbubble, params);
	}

	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {
		Log.d(TAG, "onStartCommand !");
		// return START_NOT_STICKY;
		return START_STICKY;
	}

	@Override
	public void onDestroy() {
		super.onDestroy();
		Log.d(TAG, "onDestroy !");
		// if (RecognizeResultService.mRecognizeResultService != null) {
		// this.stopService(new Intent(this, RecognizeResultService.class));
		// }
		// stop
//		if (mHandler != null) {
//			mHandler.obtainMessage(MsgType.STOP).sendToTarget();
//		}
		
		if (mHandler != null) {
			mHandler.shutDown();
		}

		try {
			Thread.sleep(2000);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		// Log.d(TAG, "killProcess myPid!");
		// android.os.Process.killProcess(android.os.Process.myPid()); //
		// 进程捕获到异常后将自己关闭
	}

	private String PaddingVersionCodeto4Char(int i) {
		if (i < 10)
			return "000" + Integer.toString(i);
		else if (i < 100)
			return "00" + Integer.toString(i);
		else if (i < 1000)
			return "0" + Integer.toString(i);
		else
			return Integer.toString(i);
	}

	@Override
	public IBinder onBind(Intent arg0) {
		// TODO Auto-generated method stub
		return null;
	}

	public void cleardailyLog() {
		Log.d(TAG, "cleardailyLog");

		try {
			String dailyLogPath = getFilesDir() + "/dailyLog";
			File dailyLogFile = new File(dailyLogPath);
			if (dailyLogFile.isDirectory()) {
				File[] dailyLogFiles = dailyLogFile.listFiles();
				Log.d(TAG, "dailyLogFiles.length->" + dailyLogFiles.length);

				if (dailyLogFiles.length > 0) {
					for (int i = 0; i < dailyLogFiles.length; i++) {
						Log.d(TAG, "dailyLogFiles[i].getPath()" + dailyLogFiles[i].getPath());

						if (dailyLogFiles[i].exists()) {
							Log.d(TAG, "dailyLogFiles[i].getName()->" + dailyLogFiles[i].getName());

							String fileCode = dailyLogFiles[i].getName()
									.substring(dailyLogFiles[i].getName().lastIndexOf(".") + 1);
							Log.d(TAG, "fileCode->" + fileCode);

							int fileCodeInt = 0;
							try {
								fileCodeInt = Integer.parseInt(fileCode);
							} catch (Exception e) {
								e.printStackTrace();

								// write log
							}

							Log.d(TAG, "fileCodeInt->" + fileCodeInt);
							// log文件数目，默认最多3个，测试apk可以为10个
							if (fileCodeInt > 10 - 1) {
								boolean deleteState = dailyLogFiles[i].delete();
								Log.d(TAG, "deleteState->" + deleteState);
							}
						} else {
							Log.d(TAG, dailyLogFiles[i].getPath() + " is not exists");
						}
					}
				}
			} else {
				Log.d(TAG, dailyLogPath + " is no isDirectory");
			}
		} catch (Exception e) {
			e.printStackTrace();

			// write log
		}
	}

}
