package com.tcl.recognize.tslite;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.tcl.recognize.util.Constant;
import com.tcl.weilong.mace.MaceClassifier;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

//import com.tcl.weilong.mace.MaceClassifier;

public class LocalRecognize {
	private static final String TAG = LocalRecognize.class.getSimpleName();
	public static LocalRecognize mLocalRecgnize;
	public ImageClassifier imageClassifier;
	private boolean initFlat = false;
    private MaceClassifier maceClassifier;

	// private static Context mcontext ;
	public static LocalRecognize getInstance(Context context) {
		Log.d(TAG, "mLocalRecgnize->" + mLocalRecgnize);
		// mcontext = context;
		if (mLocalRecgnize == null) {
			mLocalRecgnize = new LocalRecognize();
		}

		return mLocalRecgnize;
	}

	public boolean init(Context context) {
		try {
//			copyFile(context);

			try {
				if (Constant.MaceOrTflite){
					imageClassifier = new ImageClassifier(context);
				}else{
					maceClassifier = new MaceClassifier();
				}
			} catch (Exception e) {
				e.printStackTrace();
				return false;
			}
		} catch (Exception e) {
			// TODO: handle exception
			Log.e(TAG, "initTensorflow error");
			return false;
		}
		return true;
	}


	// 模型计算
	public String classifyFrame(Context context, Bitmap bitmap, String imagePath, long starttime) {
		String result = "";
		if (Constant.MaceOrTflite){
			if (imageClassifier != null) {
				result = imageClassifier.classifyFrame(context, bitmap, imagePath, starttime);
			}
		}else{
			if (maceClassifier != null) {
				result = maceClassifier.classifyFrame(context, bitmap, imagePath, starttime);
			}
		}
		return result;
	}

	private void copyFile(Context context) {
		Log.d(TAG, "copyFile");

		// try {
		// Log.d(TAG, "copyFile start-labels.txt");
		//
		// String fileName = context.getFilesDir() + "/" + Constant.LABEL_PATH;
		// InputStream assetsFile;
		// // File file = new File(fileName);
		// // if (!file.exists()) {
		// // LogManagerUtil.d(TAG, fileName + " file.exists() is false");
		//
		// try {
		// assetsFile = context.getAssets().open(Constant.LABEL_PATH);
		// OutputStream fileOut = new FileOutputStream(fileName);
		// byte[] buffer = new byte[1024];
		// int length;
		// while ((length = assetsFile.read(buffer)) > 0) {
		// fileOut.write(buffer, 0, length);
		// }
		//
		// fileOut.flush();
		// fileOut.close();
		// assetsFile.close();
		//
		// Log.d(TAG, "copyFile end-labels.txt");
		// } catch (IOException e) {
		// e.printStackTrace();
		// }
		// // } else {
		// // LogManagerUtil.d(TAG, fileName + " file.exists() is true");
		// // }
		// } catch (Exception e) {
		// e.printStackTrace();
		// }

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

}