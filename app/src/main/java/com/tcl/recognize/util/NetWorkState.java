package com.tcl.recognize.util;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;

public class NetWorkState {

	private Context mContext;

	public NetWorkState(Context context) {
		mContext = context;
	}

	public boolean isNetworkConnected() {

		if (mContext != null) {
			ConnectivityManager mConnectivityManager = (ConnectivityManager) mContext
					.getSystemService(Context.CONNECTIVITY_SERVICE);
			NetworkInfo mNetworkInfo = mConnectivityManager
					.getActiveNetworkInfo();
			if (mNetworkInfo != null) {

				return mNetworkInfo.isAvailable();
			}
		}

		// EthernetManager mEthernetManager = EthernetManager.getInstance();
		// return mEthernetManager.IsNetworkConnected();

		return false;
	}
}
