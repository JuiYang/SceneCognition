package com.tcl.recognize.tv;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.CompressFormat;
import android.graphics.BitmapFactory;
import android.os.Environment;
import android.os.Handler;
import android.os.StatFs;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicResize;
import android.renderscript.Type;
import android.util.Log;

import com.tcl.recognize.tslite.LocalRecognize;
import com.tcl.recognize.tv.ServiceHandler.MsgType;
import com.tcl.recognize.util.Constant;
import com.tcl.recognize.util.DeviceInfo;
import com.tcl.recognize.util.LogManagerUtil;
import com.tcl.recognize.util.TimestampUtil;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class RecognizeTask implements Runnable {
	public static final String TAG = "RecognizeTask";

	public static RecognizeTask mRecognizeTask;
	private TimestampUtil timestampUtil;// = new TimestampUtil();
	public static volatile boolean hasNextScreenshot = false;
	/*
	 * error string
	 */
	public static String errString = "";
	/*
	 * fields
	 */
	private String mTag;
	private Handler mHandler;
	private Context mContext;

	public String identify_id;
	private LocalRecognize mLocalRecgnize;
	private LocalResize localResize;



	public RecognizeTask(String tag, Handler mHandler, Context context) {
		mRecognizeTask = this;
		this.mTag = tag;

		this.mHandler = mHandler;
		this.mContext = context;

	}

	@Override
	public void run() {
		Log.i(TAG, "start...(trigged by " + this.mTag + ")");
		timestampUtil = new TimestampUtil();
		Long screenshot_time = null;
		// add by sunxz for Bug:0071718 20160520 below
		File path = Environment.getDataDirectory();
		StatFs stat = new StatFs(path.getPath());
		long blockSize = stat.getBlockSize();
		long availableBlocks = stat.getAvailableBlocks();
		long size = availableBlocks * blockSize;
		Log.i(TAG, "data size MB :" + size / 1024 / 1024);

		if (size / 1024 / 1024 <= 100) {
			Log.i(TAG, "data size less than ...");
			return;
		}

		/*
		 * request server for recognize
		 */
		// get parameters

		identify_id = DeviceInfo.mDeviceInfo.getIdentifyId();

		hasNextScreenshot = true;

		// 截图,台标识别，完成在把图片转成byte[],进行网络识别

		screenshot_time = System.currentTimeMillis();
		Log.d(TAG, "screenshot_time = " + screenshot_time);

		long startTime = SystemClock.uptimeMillis();
		Boolean screenshotOk = ScreenshotNewPlatform.getInstance().getScreenshotPictrue(mContext);
		long endTime0 = SystemClock.uptimeMillis();
		String timeTakePic = Long.toString(endTime0 - startTime);
		LogManagerUtil.d(TAG, "Timecost to capture screen : " + timeTakePic);
		
		String result = "wwww";
		// 截图完成
		if (screenshotOk) {
			long startRecognizeTime = SystemClock.uptimeMillis();
			// 模型检测
			result = startRecognize(startTime);
			Log.d(TAG, "local recognize result == " + result);
			long endRecognizeTime = SystemClock.uptimeMillis();
			String timeRecognize = Long.toString(endRecognizeTime - startRecognizeTime);
			LogManagerUtil.d(TAG, "Timecost to run recognize : " + timeRecognize);
			Log.i(TAG, "Timecost to run recognize : " + timeRecognize);
		}
		long endTime = SystemClock.uptimeMillis();
		String timeAll = Long.toString(endTime - startTime);
		LogManagerUtil.d(TAG, "Timecost to run all : " + timeAll);
		result = result + timeAll + "ms";
			// 发送结果
			onSuccess(result);

		hasNextScreenshot = false;

		// }

		return;
	}

	private void onSuccess(String result) {

		if (result == null) {
			return;
		}

		// 发消息
		mHandler.obtainMessage(MsgType.RECOGNIZE_RESULT, result).sendToTarget();

	}

	private void onFail(String errMsg) {
		hasNextScreenshot = false;

		mHandler.obtainMessage(MsgType.RECOGNIZE_ERROR, errMsg).sendToTarget();
	}

	public String startRecognize(long starttime) {
		// tmp/SceneDetection/screenshot.jpg
		Log.d(TAG, "startRecognize");
		String recognizeResult = "default";
		try {
			localResize = LocalResize.getInstance(mContext);
			mLocalRecgnize = LocalRecognize.getInstance(mContext);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

		if (mLocalRecgnize != null) {
			try {

				long startTime = SystemClock.uptimeMillis();
				String imageAbsPath = saveImg();
				long endTime = SystemClock.uptimeMillis();

				Log.d(TAG, "Timecost to resize screencap img: " + Long.toString(endTime - startTime));
				Log.d(TAG, "startRecognize imageAbsPath == " + imageAbsPath);
				Bitmap bitmap = BitmapFactory.decodeFile(imageAbsPath);

				Log.d(TAG, "bitmap->" + bitmap);

				if (bitmap != null) {
					Log.d(TAG, "imageClassifier.classifyFrame(bitmap)");

					recognizeResult = mLocalRecgnize.classifyFrame(mContext, bitmap, imageAbsPath, starttime);
					Log.d(TAG, "local regnize result == " + (imageAbsPath + "::" + recognizeResult));
					bitmap.recycle();
				} else {
					recognizeResult = "default";
				}
			} catch (Exception e) {
				e.printStackTrace();
			}

		}
		Log.d(TAG, "recognizeResult->" + recognizeResult);
		return recognizeResult;
	}

	private String saveImg() {
		String SAVE_PATH = mContext.getFilesDir() + File.separator + "Img";
		String filePath = "";
		Log.d(TAG, "saveImg SAVE_PATH == " + SAVE_PATH);
		File saveFile = new File(SAVE_PATH);
		if (saveFile.isDirectory() == false) {
			saveFile.mkdirs();// 创建下载目录
		} else if (saveFile.isDirectory()) {
			File[] imgFiles = saveFile.listFiles();
			Log.d(TAG, "imgFiles.length->" + imgFiles.length);
			if (imgFiles.length > 10000) {
				for (int i = 0; i < imgFiles.length; i++) {
					File f = imgFiles[i];
					f.delete();
				}
			}
		}
		String imageSC = Constant.SAVE_PATH + File.separator + Constant.SAVE_FILE;
		String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmssSSS").format(new Date());
		// whj 2018-09-05
		if (Constant.DEBUG) {
			filePath = SAVE_PATH + "/" + "IMG_" + timeStamp + ".jpg";
		} else {
			filePath = SAVE_PATH + "/" + "IMG" + ".jpg";
		}

		rootFilePath(SAVE_PATH, filePath);
		Log.d(TAG, "saveImg filePath == " + filePath);
		if (localResize != null) {
			Log.d(TAG, "saveImg localResize ----- ");
			localResize.Resize(imageSC, Constant.INPUTWIDTH, filePath);
		} else {
			Log.d(TAG, "saveImg android api resize ----- ");
			Bitmap bitmap = BitmapFactory.decodeFile(imageSC);
			if (bitmap != null) {
				Bitmap cutImgBitmap = Bitmap.createScaledBitmap(bitmap, Constant.INPUTWIDTH, Constant.INPUTHEIGHT,
						true);
				try {
					File file = new File(filePath);
					FileOutputStream resultfos = new FileOutputStream(file);
					BufferedOutputStream resultbos = new BufferedOutputStream(resultfos);
					if (cutImgBitmap != null) {
						// whj 2018-07-09
						cutImgBitmap.compress(CompressFormat.JPEG, 100, resultbos);
					}
					resultfos.flush();
					resultbos.close();
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
			bitmap.recycle();
		}

		// Bitmap croppedBitmap = BitmapFactory.decodeFile(imageSC);
		// Bitmap bitmap = rsResize(mContext, croppedBitmap);
		// if (bitmap != null) {
		// Bitmap cutImgBitmap = Bitmap.createScaledBitmap(bitmap,
		// bitmap.getWidth(), bitmap.getHeight(),
		// true);
		// try {
		// File file = new File(filePath);
		// FileOutputStream resultfos = new FileOutputStream(file);
		// BufferedOutputStream resultbos = new BufferedOutputStream(resultfos);
		// if (cutImgBitmap != null) {
		// // whj 2018-07-09
		// cutImgBitmap.compress(CompressFormat.JPEG, 100, resultbos);
		// }
		// resultfos.flush();
		// resultbos.close();
		// } catch (Exception e) {
		// e.printStackTrace();
		// }
		// }
		// bitmap.recycle();

		//// whj 2018-08-17
		// String filePathtest = "";
		// Bitmap bitmaptest = BitmapFactory.decodeFile(imageSC);
		// if (bitmaptest != null) {
		// Bitmap cutImgBitmap = Bitmap.createScaledBitmap(bitmaptest,
		// bitmaptest.getWidth(), bitmaptest.getHeight(), true);
		// try {
		//
		// // whj 2018-07-09
		// filePathtest = "/tmp" + File.separator + "IMG_" + timeStamp + ".jpg";
		// Log.d(TAG, "saveImg filePath == " + filePathtest);
		// rootFilePath("/tmp", filePathtest);
		//
		// File file = new File(filePathtest);
		// FileOutputStream resultfos = new FileOutputStream(file);
		// BufferedOutputStream resultbos = new BufferedOutputStream(resultfos);
		// if (cutImgBitmap != null) {
		// // whj 2018-07-09
		// cutImgBitmap.compress(CompressFormat.JPEG, 100, resultbos);
		// }
		// resultfos.flush();
		// resultbos.close();
		// } catch (Exception e) {
		// e.printStackTrace();
		// }
		// }
		// bitmaptest.recycle();

		return filePath;
	}

	private void rootFilePath(String path, String file) {
		Runtime runtime = Runtime.getRuntime();
		try {
			runtime.exec("chmod -R 777 " + path);
			runtime.exec("chmod -R 777 " + file);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	public static Bitmap rsResize(Context context, Bitmap image) {
		Log.d(TAG, "rsResize start");
		RenderScript rs = RenderScript.create(context);
		// final int width = (int) (image.getWidth() * scale);
		// final int height = (int) (image.getHeight() * scale);
		Bitmap outputBitmap = Bitmap.createBitmap(Constant.INPUTWIDTH, Constant.INPUTHEIGHT, Bitmap.Config.ARGB_8888);
		Allocation in = Allocation.createFromBitmap(rs, image);
		Type t = Type.createXY(rs, in.getElement(), Constant.INPUTWIDTH, Constant.INPUTHEIGHT);
		Allocation tmp1 = Allocation.createTyped(rs, t);
		// 缩放
		ScriptIntrinsicResize theIntrinsic = ScriptIntrinsicResize.create(rs);
		theIntrinsic.setInput(in);
		theIntrinsic.forEach_bicubic(tmp1);
		tmp1.copyTo(outputBitmap);
		image.recycle();
		rs.destroy();

		return outputBitmap;
	}

}
