/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.



Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at



    http://www.apache.org/licenses/LICENSE-2.0



Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.

==============================================================================*/

package com.tcl.recognize.tslite;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.renderscript.Allocation;
import android.renderscript.RenderScript;
import android.renderscript.ScriptIntrinsicResize;
import android.renderscript.Type;
import android.util.Log;

import com.tcl.recognize.util.Constant;
import com.tcl.recognize.util.LogManagerUtil;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.NativeInterpreterWrapper;
import org.tensorflow.lite.TensorFlowLite;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/** Classifies images with Tensorflow Lite. */

public class ImageClassifier {
	private static final String TAG = ImageClassifier.class.getSimpleName();

	// /** Tag for the {@link Log}. */
	//

	/** Name of the model file stored in Assets. */

	private static final String MODEL_PATH = Constant.MODEL_PATH;

	/** Name of the label file stored in Assets. */

	private static final String LABEL_PATH = Constant.LABEL_PATH;

	/** Number of results to show in the UI. */

	private static final int RESULTS_TO_SHOW = 3;

	/** Dimensions of inputs. */

	private static final int DIM_BATCH_SIZE = 1;

	private static final int DIM_PIXEL_SIZE = 3;

	static final int DIM_IMG_SIZE_X = Constant.INPUTWIDTH;

	static final int DIM_IMG_SIZE_Y = Constant.INPUTHEIGHT;

	/* Preallocated buffers for storing image data in. */

	private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

	/**
	 * An instance of the driver class to run model inference with Tensorflow
	 * Lite.
	 */

	private Interpreter tflite;

	/** Labels corresponding to the output of the vision model. */

	private List<String> labelList;

	/**
	 * A ByteBuffer to hold image data, to be feed into Tensorflow Lite as
	 * inputs.
	 */

	private ByteBuffer imgData = null;
	// private float[] imgData = new float[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * 3];

	/**
	 * An array to hold inference results, to be feed into Tensorflow Lite as
	 * outputs.
	 */
	// ningyb 20180622
	// private byte[][] labelProbArrayByte = null;
	// private float[][] filterLabelProbArray = null;
	// private static final int FILTER_STAGES = 3;
	// // private static final float FILTER_FACTOR = 0.4f;
	// private static final int numBytesPerChannel = 1;

	private float[][] labelProbArrayFloat = null;
	private float[][] filterLabelProbArray = null;
	private static final int FILTER_STAGES = 3;
	// private static final float FILTER_FACTOR = 0.4f;
	private static final int numBytesPerChannel = 4;

	/**
	 * The inception net requires additional normalization of the used input.
	 */
	private static final int IMAGE_MEAN = 128;
	private static final float IMAGE_STD = 128.0f;


	private PriorityQueue<Map.Entry<String, Float>> sortedLabels =

			new PriorityQueue<>(RESULTS_TO_SHOW, new Comparator<Map.Entry<String, Float>>() {

				@Override
				public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {

					return (o1.getValue()).compareTo(o2.getValue());
				}
			});

	/** Initializes an {@code ImageClassifier}. */

	public ImageClassifier(Context context) throws IOException {
		Log.d(TAG, "ImageClassifier");

		String version = TensorFlowLite.version();
		Log.d(TAG, "TF-version->" + version);

		// tflite = new Interpreter(loadModelFile(context));

		// nignyb 20171218
		String filePath = context.getFilesDir() + "/" + MODEL_PATH;
		File modelFile = new File(filePath);
		tflite = new Interpreter(modelFile);
		tflite.setNumThreads(4);

		labelList = loadLabelList(context);
		Log.d(TAG, "labelList.size()->" + labelList.size());

		imgData = ByteBuffer
				.allocateDirect(DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE * numBytesPerChannel);

		imgData.order(ByteOrder.nativeOrder());

		// ningyb 20180622
		// labelProbArrayByte = new byte[1][getNumLabels()];
		labelProbArrayFloat = new float[1][getNumLabels()];
		// filterLabelProbArray = new float[FILTER_STAGES][getNumLabels()];
		// Log.d(TAG,
		// "NativeInterpreterWrapper.dataTypeOf(filterLabelProbArray)-init->"
		// + NativeInterpreterWrapper.dataTypeOf(filterLabelProbArray));

		Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
	}

	/** Classifies a frame from the preview stream. */

	public String classifyFrame(Context context, Bitmap croppedBitmap, String imagePath, long starttime) {

		if (tflite == null) {
			Log.e(TAG, "Image classifier has not been initialized; Skipped.");

			return "Uninitialized Classifier.";
		}

		Log.d(TAG, "recognizeImage start");
		long startTime0 = SystemClock.uptimeMillis();
		LogManagerUtil.d(TAG, "imagePath is = " + imagePath);

		// 图片尺度变化
		Bitmap bitmap = Bitmap.createScaledBitmap(croppedBitmap,
				Constant.INPUTWIDTH, Constant.INPUTHEIGHT, false);


		// resize to 224x224

		// Bitmap bitmap = Bitmap.createScaledBitmap(croppedBitmap,
		// Constant.INPUTWIDTH, Constant.INPUTHEIGHT, false);

		// Bitmap bitmap = rsResize(context, croppedBitmap);
		Log.d(TAG, "recognizeImage start1");

		// long endResizeTime = SystemClock.uptimeMillis();
		// Log.d(TAG, "Timecost to resize bitmap: " +
		// Long.toString(endResizeTime - startTime0));

		convertBitmapToByteBuffer(bitmap);


		// Here's where the magic happens!!!

		long startTime = SystemClock.uptimeMillis();

		Log.d(TAG, "NativeInterpreterWrapper.dataTypeOf(imgData)->" + NativeInterpreterWrapper.dataTypeOf(imgData));

		// ningyb 20180622
		// Log.d(TAG,
		// "NativeInterpreterWrapper.dataTypeOf(labelProbArrayByte)->"
		// + NativeInterpreterWrapper.dataTypeOf(labelProbArrayByte));
		//
		// tflite.run(imgData, labelProbArrayByte);

//		Log.d(TAG, "NativeInterpreterWrapper.dataTypeOf(labelProbArrayFloat)->"
//				+ NativeInterpreterWrapper.dataTypeOf(labelProbArrayFloat));

		tflite.run(imgData, labelProbArrayFloat);

		long endTime = SystemClock.uptimeMillis();

		String costTime = Long.toString(endTime - starttime);
		Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

		// // ningyb 20180622 Smooth the results across frames.
		// applyFilter();

		String textToShow = printTopKLabels();

		if (!Constant.DEBUG) {
			sendBroadCast(context, textToShow, costTime);
		}

		// whj 2018-09-05
		if (Constant.DEBUG) {
			String recognizeResult = textToShow.substring(0, textToShow.indexOf(":"));
			Log.d(TAG, "recognizeResult ====== " + recognizeResult);
			Log.d(TAG, "imagePath ====== " + imagePath);
			String newImagePath = imagePath.substring(0, imagePath.lastIndexOf("."));
			Log.d(TAG, "newImagePath ====== " + newImagePath);
			renameFile(imagePath, newImagePath + "_" + recognizeResult + ".jpg");
		}

		long endTime0 = SystemClock.uptimeMillis();
		String timeRunTs = Long.toString(endTime0 - startTime0);
		LogManagerUtil.d(TAG, "Timecost to run ts is : " + timeRunTs);
		textToShow = textToShow + "\n" + timeRunTs + "ms/";

		Log.d(TAG, "recognizeImage end");
		croppedBitmap.recycle();

		return textToShow;
	}


	// // ningyb 20180622
	// public void applyFilter() {
	// LogManagerUtil.d(TAG, "applyFilter");
	//
	// int numLabels = getNumLabels();
	// LogManagerUtil.d(TAG, "numLabels->" + numLabels);
	//
	// // Low pass filter `labelProbArray` into the first stage of the filter.
	// for (int j = 0; j < numLabels; ++j) {
	//// filterLabelProbArray[0][j] += FILTER_FACTOR * (labelProbArrayByte[0][j]
	// - filterLabelProbArray[0][j]);
	// filterLabelProbArray[0][j] += FILTER_FACTOR * (labelProbArrayFloat[0][j]
	// - filterLabelProbArray[0][j]);
	// }
	// // Low pass filter each stage into the next.
	// for (int i = 1; i < FILTER_STAGES; ++i) {
	// for (int j = 0; j < numLabels; ++j) {
	// filterLabelProbArray[i][j] += FILTER_FACTOR
	// * (filterLabelProbArray[i - 1][j] - filterLabelProbArray[i][j]);
	// }
	// }
	//
	// // Copy the last stage filter output back to `labelProbArray`.
	// for (int j = 0; j < numLabels; ++j) {
	// // labelProbArrayByte[0][j] = (byte)
	// // filterLabelProbArray[FILTER_STAGES - 1][j];
	// labelProbArrayFloat[0][j] = filterLabelProbArray[FILTER_STAGES - 1][j];
	// }
	// }

	protected int getNumLabels() {
		return labelList.size();
	}

	/** Closes tflite to release resources. */

	public void close() {

		tflite.close();

		tflite = null;
	}

	/** Reads label list from Assets. */

	private List<String> loadLabelList(Context context) throws IOException {

		List<String> labelList = new ArrayList<String>();

		BufferedReader reader = new BufferedReader(new InputStreamReader(context.getAssets().open(LABEL_PATH)));

		String line;

		while ((line = reader.readLine()) != null) {
			labelList.add(line);
		}

		reader.close();
		return labelList;
	}

	/** Memory-map the model file in Assets. */
	@SuppressWarnings({ "resource", "unused" })
	private MappedByteBuffer loadModelFile(Context context) throws IOException {

		AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_PATH);
		FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
		FileChannel fileChannel = inputStream.getChannel();
		long startOffset = fileDescriptor.getStartOffset();
		long declaredLength = fileDescriptor.getDeclaredLength();

		return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);

		// // nignyb 20171218
		// String filePath = context.getFilesDir() + "/" + MODEL_PATH;
		// FileOutputStream fileOutputStream = new FileOutputStream(filePath);
		// FileDescriptor fileDescriptor = fileOutputStream.getFD();
		// FileInputStream inputStream = new FileInputStream(fileDescriptor);
		// FileChannel fileChannel = inputStream.getChannel();
		// long size = fileChannel.size();
		//
		// return fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, size);
	}

	/** Writes Image data into a {@code ByteBuffer}. */

	private void convertBitmapToByteBuffer(Bitmap bitmap) {

		if (imgData == null) {
			return;
		}

		imgData.rewind();

		bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

		// Convert the image to floating point.

		int pixel = 0;

		long startTime = SystemClock.uptimeMillis();

		for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
			for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
				final int val = intValues[pixel++];

				// ningyb 20180622
				// imgData.put((byte) ((val >> 16) & 0xFF));
				// imgData.put((byte) ((val >> 8) & 0xFF));
				// imgData.put((byte) (val & 0xFF));

				imgData.putFloat((float) ((val >> 16) & 0xFF) / 255);
				imgData.putFloat((float) ((val >> 8) & 0xFF) / 255);
				imgData.putFloat((float) (val & 0xFF) / 255);
			}
		}

		// for (int i = 0; i < intValues.length; ++i) {
		// final int val = intValues[i];
		// imgData[i * 3 + 0] = (float) ((((val >> 16) & 0xFF) / 255 - 0.485) /
		// 0.229);
		// imgData[i * 3 + 1] = (float) ((((val >> 8) & 0xFF) / 255 - 0.456) /
		// 0.224);
		// imgData[i * 3 + 2] = (float) (((val & 0xFF) / 255 - 0.406) / 0.225);
		// // LogManagerUtil.d(TAG, inputSingle[i * 3 + 0] + "/" +
		// // inputSingle[i * 3 + 1] + "/" + inputSingle[i * 3 + 2]);
		// }

		long endTime = SystemClock.uptimeMillis();

		Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
	}

	/** Prints top-K labels, to be shown in UI as the results. */

	private String printTopKLabels() {
		Log.d(TAG, "printTopKLabels");

		for (int i = 0; i < labelList.size(); ++i) {
			// Log.d(TAG, "(labelProbArrayByte[0][i] & 0xff)->" +
			// (labelProbArrayByte[0][i] & 0xff));
			// Log.d(TAG, "labelProbArrayFloat[0][i]->" +
			// labelProbArrayFloat[0][i]);

			sortedLabels.add(
					// ningyb 20180622
					// new AbstractMap.SimpleEntry<>(labelList.get(i),
					// (labelProbArrayByte[0][i] & 0xff) / 255.0f)
					new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArrayFloat[0][i]));

			if (sortedLabels.size() > RESULTS_TO_SHOW) {
				// Log.d(TAG, "sortedLabels.size()->" + sortedLabels.size());

				sortedLabels.poll();// poll:相当于先get然后再remove掉,就是查看的同时,也将这个元素从容器中删除掉
			}
		}

		String textToShow = "";

		final int size = sortedLabels.size();
		Log.d(TAG, "size->" + size);

		float value = 0;

		for (int i = 0; i < size; ++i) {
			// Log.d(TAG, "i->" + i);

			Map.Entry<String, Float> label = sortedLabels.poll();

			// textToShow = "\n" + label.getKey() + ":" +
			// Float.toString(label.getValue()) + textToShow;

			LogManagerUtil.d(TAG, "label.getKey() + : + Float.toString(label.getValue()->" + label.getKey() + ":"
					+ Float.toString(label.getValue()));

			if (value < label.getValue()) {
				value = label.getValue();
				textToShow = label.getKey() + ":" + Float.toString(label.getValue());
			}
		}

		return textToShow;
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


	/**
	 * 重命名文件
	 * 
	 * @param oldPath
	 *            原来的文件地址
	 * @param newPath
	 *            新的文件地址
	 */
	public static void renameFile(String oldPath, String newPath) {
		Log.d(TAG, "renameFile oldPath == " + oldPath);
		Log.d(TAG, "renameFile newPath == " + newPath);
		File oleFile = new File(oldPath);
		File newFile = new File(newPath);
		// 执行重命名
		oleFile.renameTo(newFile);
	}

	private void sendBroadCast(Context context, String output, String costTime) {
		Log.d(TAG, "sendBroadCast output =========== " + output);
		String resultKey = "";
		String degree = "";
		String result = "";
		String keyWord = "";
		try {
			resultKey = output.substring(0, output.lastIndexOf(":"));
			degree = output.substring(output.lastIndexOf(":") + 1);
			result = resultKey.substring(0, resultKey.indexOf(":"));
			keyWord = resultKey.substring(resultKey.indexOf(":") + 1);
		} catch (Exception e) {
			// TODO: handle exception
			e.printStackTrace();
		}

		Log.d(TAG, "sendBroadCast result and key =========== " + resultKey);
		Log.d(TAG, "sendBroadCast degree =========== " + degree);
		Log.d(TAG, "sendBroadCast result =========== " + result);
		Log.d(TAG, "sendBroadCast keyWord =========== " + keyWord);
		String messageforThirdBackBroadcast = Constant.RECOGNIZE_RESULT_BRO;
		Intent messageforThirdIntent = new Intent(messageforThirdBackBroadcast);
		messageforThirdIntent.putExtra("costTime", costTime);
		messageforThirdIntent.putExtra("keyWord", keyWord);
		messageforThirdIntent.putExtra("result", result);
		messageforThirdIntent.putExtra("degree", degree);
		messageforThirdIntent.setFlags(Intent.FLAG_INCLUDE_STOPPED_PACKAGES);
		context.getApplicationContext().sendBroadcast(messageforThirdIntent);
	}

}