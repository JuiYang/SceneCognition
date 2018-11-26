package com.tcl.recognize.tv;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Handler;
import android.os.Message;
import android.util.Log;

import com.tcl.recognize.util.Constant;
import com.tcl.recognize.util.FlashFreeSpace;

public class StartupReceiver extends BroadcastReceiver {
	private static final String TAG = "StartupReceiver";
	private static final String STARTANDEXIT = "com.android.tcl.messagebox.MessageforThird.InputSource";

	private Context mContext;
	
	private Handler handler = new Handler() {
		public void handleMessage(Message msg) {
			handleMessages(msg);
		}
	};

	@Override
	public void onReceive(Context context, Intent intent) {
		Log.d(TAG, "onReceive " + intent.getAction());
		String result = intent.getStringExtra("result");
		String degree = intent.getStringExtra("degree");
		Log.d(TAG, "onReceive result = " + result);
		Log.d(TAG, "onReceive degree = " + degree);

		FlashFreeSpace ffs = new FlashFreeSpace();
		mContext = context;
//		GetConfig getConf = new GetConfig(handler, context);
//		Constant.DEBUG = getConf.getDebugMode();
		Log.d(TAG, "Constant.DEBUG == " +Constant.DEBUG);
		if (Constant.DEBUG) {
			if (intent.getAction().equals(Intent.ACTION_BOOT_COMPLETED)) {
				if (!ffs.isEnough()) {
					Log.d(TAG, "FlashFreeSpace.isEnough() = " + ffs.isEnough());
					return;
				}
				context.startService(new Intent(context, RecognizeService.class));
			}
		} else {
			if (intent.getAction().equals(Constant.START_SEVVICE_BRO)) {
				if (!ffs.isEnough()) {
					Log.d(TAG, "FlashFreeSpace.isEnough() = " + ffs.isEnough());
					return;
				}
				context.startService(new Intent(context, RecognizeService.class));
			}

			if (intent.getAction().equals(Constant.STOP_SEVVICE_BRO)) {
				if (!ffs.isEnough()) {
					Log.d(TAG, "FlashFreeSpace.isEnough() = " + ffs.isEnough());
					return;
				}
				// context.stopService(new Intent(context,
				// RecognizeService.class));
				StopServiceThread mStopServiceThread = new StopServiceThread();
				mStopServiceThread.start();
			}
		}

	}

	class StopServiceThread extends Thread {
		public void run() {
			Log.d(TAG, "StopService");
			// android.os.Process.killProcess(android.os.Process.myPid());
			mContext.stopService(new Intent(mContext, RecognizeService.class));

			try {
				Thread.sleep(2000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}

			System.exit(0);

		}
	}
	
	
	private void handleMessages(Message msg) {
		switch (msg.what) {
		default:
			break;
		}
	}

}
