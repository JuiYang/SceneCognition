package com.tcl.recognize.tv;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.util.Log;

import com.tcl.app.AppManager;
import com.tcl.app.screenshot.Screenshot;
import com.tcl.recognize.util.Constant;
import com.tcl.recognize.util.ErrorCode;
import com.tcl.tvmanager.TTv3DManager;
import com.tcl.tvmanager.vo.EnTCL3DVideoDisplayFormat;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;

//import java.sql.Date;

public class ScreenshotNewPlatform {
	private static final String TAG = "ScreenshotNewPlatform";

	static ScreenshotNewPlatform mScreenshotNewPlatform;
	private static final int ORIGINAL_WIDTH = 1920;
	private static final int ORIGINAL_HEIGHT = 1080;

	private static final int ORIGINAL_WIDTH_32 = 1280;
	private static final int ORIGINAL_HEIGHT_32 = 720;

	// private Context mContext;

	// private Bitmap bgBitmap;
	int bgWidth;
	int bgHeight;
	int imgX_center;
	int imgY_center;
	int img_width;
	int img_height;
	static String feature;

	// for Rt95
	private static final String SAVE_PATH = Constant.SAVE_PATH;
	private static final String SAVE_FILE = Constant.SAVE_FILE;
	private AppManager mAppManager;

	private ByteBuffer bBuffer = null;
	public String errString = null;
	public static String logoName;
	private String mDevModel = "";

	public ScreenshotNewPlatform() {
		super();

		Log.d(TAG, "ScreenshotNewPlatform");

	}

	public static ScreenshotNewPlatform getInstance() {

		if (mScreenshotNewPlatform == null) {
			mScreenshotNewPlatform = new ScreenshotNewPlatform();
		}
		return mScreenshotNewPlatform;
	}

	private void rootFilePath() {
		String pathname = SAVE_PATH + File.separator + SAVE_FILE;
		Runtime runtime = Runtime.getRuntime();
		try {
			runtime.exec("chmod -R 777 " + SAVE_PATH);
			runtime.exec("chmod -R 777 " + pathname);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	/**
	 * 执行截图接口，以 bitmap
	 * 
	 * @param context
	 * @author fanfan
	 * @return
	 */
	public boolean getScreenshotPictrue(Context context) {
		Log.d(TAG, "getScreenshotPictrue!");

		this.mAppManager = AppManager.getInstance(context);

		if (EnTCL3DVideoDisplayFormat.EN_TCL_3D_NONE != TTv3DManager.getInstance(context).getDisplayFormat()) {
			Log.d(TAG, "！= 2D,cannot takeScreen ");
			return false;

		}

		File file = null;
		String pathname = "";
		file = new File(SAVE_PATH);
		if (!file.exists()) {
			file.mkdirs();
			Log.d(TAG, "mkdirs SAVE_PATH=" + SAVE_PATH);
		}

		 pathname = SAVE_PATH + File.separator + SAVE_FILE;

		rootFilePath();

		/*
		 * step 1:screenshot
		 */
		Log.d(TAG, "screenshotStart="
				+ new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date(System.currentTimeMillis())));

		/**
		 * rect是截屏区域，截屏最后两个参数是分辨率，截屏下层会根据分辨率和截屏区域进行缩放适配
		 */
		Rect rect = new Rect(0, 0, ORIGINAL_WIDTH, ORIGINAL_HEIGHT);
		boolean pathGetScreenshotOK = false;
		if (mAppManager != null) {
			try {

				Log.d(TAG, "mDevModel" + mDevModel);

				if (mDevModel.contains("838P")) {
					/**
					 * modify by lilan_n 20171012
					 */
					Log.d(TAG, "838P ");
					pathGetScreenshotOK = mAppManager.getScreenshot().snapshot(255, rect, pathname,
							Screenshot.MODE_OSD_VIDEO, ORIGINAL_WIDTH_32, ORIGINAL_HEIGHT_32); // 32寸
																								// 全屏1280*720
				} else {
					pathGetScreenshotOK = mAppManager.getScreenshot().snapshot(255, rect, pathname,
							Screenshot.MODE_OSD_VIDEO, ORIGINAL_WIDTH, ORIGINAL_HEIGHT); // 1920*1080
				}

				Log.d(TAG, "pathGetScreenshotOK=" + pathGetScreenshotOK + " , pathname=" + pathname + " , time="
						+ System.currentTimeMillis());
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				return false;
			}
		}

		if (pathGetScreenshotOK == false) {
			errString = ErrorCode.errString_screenshot;
			Log.d(TAG, "CaptureScreen failed");
			return false;

		}

		Log.d(TAG, "screenshotEnd="
				+ new SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS").format(new Date(System.currentTimeMillis())));

		/*
		 * step 2:compute footprint
		 */

		// bgBitmap = BitmapFactory.decodeFile(pathname);
		// bgWidth = bgBitmap.getWidth();
		// bgHeight = bgBitmap.getHeight();
		// Log.d(TAG, "bgBitmap.getHeight() = " + bgBitmap.getHeight()
		// + ", bgBitmap.getWidth() = " + bgBitmap.getWidth());
		return true;

		// 全屏图，测试使用
		// try {
		// saveMyBitmap(bgBitmap);
		// } catch (IOException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }

	}

	/**
	 * 把图片 转成byte
	 * 
	 * @param bgBitmap
	 * @return
	 */
	public byte[] getScreenPictureByte(Bitmap bgBitmap) {

		if (bgBitmap == null || bgBitmap.isRecycled()) {
			return null;
		}

		bgWidth = bgBitmap.getWidth();
		bgHeight = bgBitmap.getHeight();
		Log.d(TAG, "bgBitmap.getHeight() = " + bgBitmap.getHeight() + ", bgBitmap.getWidth() = " + bgBitmap.getWidth());

		if (null == bBuffer) {

			// Out of memory on a 8294416-byte allocation.
			bBuffer = ByteBuffer.allocate(bgBitmap.getWidth() * bgBitmap.getHeight() * 4);
		}

		bgBitmap.copyPixelsToBuffer(bBuffer);
		byte[] data = bBuffer.array();

		Log.d(TAG, "bitmapByte = " + data.length);

		return data;
	}

	// public void getpanel() {
	//
	// PanelProperty pWH =
	// TTvPictureManager.getInstance(mContext).getPanelWidthHeight();
	// int panelHeignt = pWH.height;
	// int panelWidth = pWH.width;
	// Log.d(TAG, "panelHeignt = " + panelHeignt + ",panelWidth = " +
	// panelWidth);
	//
	// }

	// public void saveMyBitmap(Bitmap mBitmap) throws IOException {
	// String absolutePath = mContext.getFileStreamPath("") + File.separator;
	//
	// File f = new File(absolutePath + "screen.jpg");
	// Log.d(TAG, "saveMyBitmap!");
	// f.createNewFile();
	// FileOutputStream fOut = null;
	// try {
	// fOut = new FileOutputStream(f);
	// } catch (FileNotFoundException e) {
	// e.printStackTrace();
	// }
	// mBitmap.compress(Bitmap.CompressFormat.JPEG, 100, fOut);
	// try {
	// fOut.flush();
	// } catch (IOException e) {
	// e.printStackTrace();
	// }
	// try {
	// fOut.close();
	// } catch (IOException e) {
	// e.printStackTrace();
	// }
	// }

}
