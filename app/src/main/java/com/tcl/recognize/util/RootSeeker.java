package com.tcl.recognize.util;

import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;

public class RootSeeker {

	private static final String TAG = "RootSeeker";
	private static final int SOCKET_PORT = 8090;
	private static final String SOCKET_IP = "127.0.0.1";

	public static int exec(String cmd) {

		// Log.d(TAG,
		// "<-----------------in exec() start --------------------->");
		Log.d(TAG, "exec cmd: ooo " + cmd);

		Socket socket;
		BufferedReader in;
		PrintWriter out;
		char[] buf = new char[256];
		int ret = 0;

		try {
			Log.d(TAG, "new Socket(SOCKET_IP, SOCKET_PORT start); ");
			socket = new Socket(SOCKET_IP, SOCKET_PORT);
			Log.d(TAG, "new Socket(SOCKET_IP, SOCKET_PORT finish); ");
			in = new BufferedReader(new InputStreamReader(
					socket.getInputStream()));
			out = new PrintWriter(socket.getOutputStream(), true);
			BufferedReader line = new BufferedReader(new InputStreamReader(
					System.in));
			out.println(cmd);

			if (in.read(buf) != -1 && new String(buf).equals(new String("ok")))
				ret = 0;
			else
				ret = -1;

			Log.d(TAG, "command execute " + ((ret == 0) ? "ok" : "false")
					+ ", return: -->" + new String(buf));

			line.close();
			out.close();
			in.close();
			socket.close();
			return 0; // FIXME
		} catch (IOException e) {
			e.printStackTrace();
			Log.d(TAG, e.toString());
		}
		return 0;
	}

	public static boolean chmod(String fileName) {
		String cmd = "chmod 777 " + fileName;
		System.out.println("cmd =" + cmd);
		try {
			// Runtime runtime = Runtime.getRuntime();
			// Process process = runtime.exec(cmd);
			Runtime.getRuntime().exec(cmd);
			return true;
		} catch (Exception e) {

			e.printStackTrace();
			return false;
		}
	}

}
