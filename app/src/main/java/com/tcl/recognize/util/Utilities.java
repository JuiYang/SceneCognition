package com.tcl.recognize.util;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class Utilities {

	private static final String TAG = "Utilities";

	/**
	 * Utility class to convert a byte array to a hexadecimal string.
	 * 
	 * @param bytes
	 *            Bytes to convert
	 * @return String, containing hexadecimal representation.
	 */
	public static String ByteArrayToHexString(byte[] bytes) {
		final char[] hexArray = { '0', '1', '2', '3', '4', '5', '6', '7', '8',
				'9', 'A', 'B', 'C', 'D', 'E', 'F' };
		char[] hexChars = new char[bytes.length * 2];
		int v;
		for (int j = 0; j < bytes.length; j++) {
			v = bytes[j] & 0xFF;
			hexChars[j * 2] = hexArray[v >>> 4];
			hexChars[j * 2 + 1] = hexArray[v & 0x0F];
		}
		return new String(hexChars);
	}

	/**
	 * Utility class to convert a hexadecimal string to a byte string.
	 * 
	 * <p>
	 * Behavior with input strings containing non-hexadecimal characters is
	 * undefined.
	 * 
	 * @param s
	 *            String containing hexadecimal characters to convert
	 * @return Byte array generated from input
	 */
	public static byte[] HexStringToByteArray(String s) {
		int len = s.length();
		byte[] data = new byte[len / 2];
		for (int i = 0; i < len; i += 2) {
			data[i / 2] = (byte) ((Character.digit(s.charAt(i), 16) << 4) + Character
					.digit(s.charAt(i + 1), 16));
		}
		return data;
	}

	// add by jiang
	@SuppressLint("NewApi")
	@SuppressWarnings("deprecation")
	public static void copyFileToDir(Context context, String fromDir,
			String filename, String toDir) throws IOException {
		Log.d(TAG, "filename = " + filename);
		InputStream in = null;
		OutputStream out = null;
		AssetManager asset = context.getAssets();

		try {
			in = asset.open(filename, AssetManager.ACCESS_UNKNOWN);

			if (in == null) {
				Log.i(TAG, "read assets file error...");
				return;
			}

			if (toDir != null && !toDir.isEmpty()) {
				out = new FileOutputStream(toDir + File.separator + filename);
			} else {
				out = context.openFileOutput(filename,
						Context.MODE_WORLD_WRITEABLE);
			}

			if (out == null) {
				Log.i(TAG, "create files folder error...");
				return;
			}

			byte[] buf = new byte[1024];
			int read;
			while ((read = in.read(buf)) != -1)
				out.write(buf, 0, read);
			in.close();
			out.flush();
			out.close();
		} catch (IOException e) {
			Log.i(TAG, "Copy function error..." + e.getMessage());
			e.printStackTrace();
			if (in != null)
				in.close();
			if (out != null)
				out.close();
		}
	}
}
