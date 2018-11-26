package com.tcl.recognize.util;

import android.os.Environment;
import android.os.StatFs;
import android.util.Log;

import java.io.File;

public class FlashFreeSpace {
	private static final String TAG = "getFlashFreeSpace";

	public FlashFreeSpace() {
		super();
		// TODO Auto-generated constructor stub
	}

	public boolean isEnough() {

		long freeSize = getFlashFreeSpace() / 1024 / 1024;
		Log.d(TAG, "flash size = " + freeSize );
		if (freeSize > 20) {
			return true;
		} else {
			return false;
		}

	}

	public static long getFlashFreeSpace() {
		long size = 0;

		try {
			File path = Environment.getDataDirectory();
			StatFs statfs = new StatFs(path.getPath());

			long block = statfs.getAvailableBlocks();
			long blocksize = statfs.getBlockSize();
			size = block * blocksize * 9 / 10;
			statfs = null;
		} catch (IllegalArgumentException e) {
			Log.d(TAG, "get flash space exception.");
		}
		Log.d(TAG, "flash size = " + size);
		return size;
	}

}
