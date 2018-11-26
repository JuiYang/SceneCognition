package com.tcl.recognize.util;

import android.app.ActivityManager;
import android.app.ActivityManager.RunningTaskInfo;
import android.content.ContentResolver;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageManager.NameNotFoundException;
import android.database.Cursor;
import android.net.Uri;
import android.text.TextUtils;
import android.util.Log;

import com.tcl.xian.StartandroidService.MyUsers;
import com.tcl.xian.StartandroidService.SqlCommon;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.NetworkInterface;
import java.net.SocketException;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;

/**
 * fanfan添加 app_name 获取apk 包名  属性
 * @author fanfan
 *
 */
public class DeviceInfo {
	private static final String TAG = "RecognizeTask";

	private String deviceId;
	private String devModel;
	private String huan_id;
	private String sv;
	private InetAddress ip;
	private String macAddr;
	private String wifiMacAddr;
	private String userId;
	private String identifyId;
	// private String version;
	private String versionName;
	private String dnum;
	private String patternlibVersion;
	private String app_name = "";
	private static Context mContext;
	public static DeviceInfo mDeviceInfo = null;
	private File file;

	private static Object lock = new Object();
	
	public static DeviceInfo getInstance(Context context) {
		
		mContext = context;
		if (mDeviceInfo == null) {
			synchronized (lock) {
				if(mDeviceInfo == null) {
					mDeviceInfo = new DeviceInfo(context);
				}
			}
		}
			
		return mDeviceInfo;
	}

	protected DeviceInfo(Context context) {

		getdevinfo();
		getMacAddress();
		getWifiMacAddress();
		getHuanId();
		getUserID(context);
		get_IdentifyId();
		getTerminalVersion();
		get_Dnum();
		// 获取终端版本号
		get_PatternlibVersion();
		// 获取定标时间
		//get_TimeScaling();
		writeToConfigParametersFile();
		
		//获取当前应用的包名
		String appName = getAppPkgName(context);
		if(!TextUtils.isEmpty(appName)) {
			setApp_name(appName);
		}
		
		Log.e(TAG, "------DeviceInfo end------");
	}
	
	 /**
     * 获取app packageName
     */
    public String getAppPkgName(Context context){
        PackageManager pm = context.getPackageManager();
        PackageInfo pi;
        try {
            pi = pm.getPackageInfo(context.getPackageName(),0);
            return pi.packageName;
        } catch (NameNotFoundException e) {
            e.printStackTrace();
        }
        return null;
    }
	
	private void writeToConfigParametersFile(){			
		file = new File(mContext.getFilesDir().getPath() + File.separator 
		              + Constant.ConfigParametersFile);
		if(!file.exists()){
			try {
				file.createNewFile();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		writeValue();					
	}

	private void writeValue(){
		FileOutputStream osFile;
		OutputStreamWriter osWriter;
		try {
			osFile = new FileOutputStream(file);
			osWriter = new OutputStreamWriter(osFile);
			StringBuffer buf = new StringBuffer();
			buf.append("deviceId=" + deviceId + "\n" + "devModel=" + devModel
					 + "\n" + "huan_id=" + huan_id + "\n" + "sv=" + sv
					 + "\n" + "ip=" + ip + "\n" + "macAddr=" + macAddr
					 + "\n" + "wifiMacAddr=" + wifiMacAddr + "\n" + "userId=" + userId
					 + "\n" + "identifyId=" + identifyId + "\n" + "versionName=" + versionName
					 + "\n" + "dnum=" + dnum + "\n" + "patternlibVersion=" + patternlibVersion);	
			String outStr = buf.toString();
			osWriter.write(outStr, 0, outStr.length());
			osWriter.close();
			osFile.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
	    } catch (IOException e) {
	    	e.printStackTrace();		
	    }			
	}
	
	private void getdevinfo() {
		String strLine = null;
		String[] arrLineString = null;
		FileReader localFileReader = null;
		// mDataMap contains 4 parameter： devid、sv、hv、devmodel
		HashMap<String, String> mDataMap = new HashMap<String, String>();
		BufferedReader localBufferedReader = null;
		try {
			localFileReader = new FileReader("/data/devinfo.txt");
			localBufferedReader = new BufferedReader(localFileReader);
			while (true) {
				strLine = localBufferedReader.readLine();
				if (strLine == null) {
					break;
				}
				arrLineString = strLine.split("=");
				if (arrLineString.length >= 2) {
					mDataMap.put(arrLineString[0], arrLineString[1]);
				}
			}
			localBufferedReader.close();
			localFileReader.close();

			for (String name : mDataMap.keySet()) {
				Log.d(TAG, "name = " + name);

				if ("devid".equals(name)) {
					setDeviceId(mDataMap.get(name).trim());
					Log.d(TAG, "deviceId = " + deviceId);
				}

				if ("devmodel".equals(name)) {
					setDevModel(mDataMap.get(name).trim());
					Log.d(TAG, "devModel = " + devModel);
				}

				if ("sv".equals(name)) {
					setSv(mDataMap.get(name).trim());
					Log.d(TAG, "sv = " + sv);
				}

			}

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	protected InetAddress getLocalInetAddress() {
		InetAddress ip = null;
		try {
			Enumeration<NetworkInterface> en_netInterface = NetworkInterface
					.getNetworkInterfaces();
			while (en_netInterface.hasMoreElements()) {
				NetworkInterface ni = (NetworkInterface) en_netInterface
						.nextElement();
				Enumeration<InetAddress> en_ip = ni.getInetAddresses();
				while (en_ip.hasMoreElements()) {
					ip = en_ip.nextElement();
					if (!ip.isLoopbackAddress()
							&& ip.getHostAddress().indexOf(":") == -1)
						break;
					else
						ip = null;
				}

				if (ip != null) {
					break;
				}
			}
		} catch (SocketException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return ip;
	}

	private void getMacAddress() /* throws UnknownHostException */{

		String strMacAddr = null;

		try {
			FileReader localFileReader = new FileReader(
					"/sys/class/net/eth0/address");
			BufferedReader localBufferedReader = new BufferedReader(
					localFileReader);
			strMacAddr = localBufferedReader.readLine();
			if (strMacAddr == null || "null".endsWith(strMacAddr)) {
				strMacAddr = Constant.unknown;
			}
			if (!Constant.unknown.endsWith(strMacAddr)) {
				strMacAddr = strMacAddr.trim();
				Log.d(TAG, "strMacAddr = " + strMacAddr);
				strMacAddr = strMacAddr.replace(":", "-").toUpperCase();
			}
			localBufferedReader.close();
			localFileReader.close();
			setMacAddr(strMacAddr.trim());
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	private void getWifiMacAddress() {
		String wifiMacAddress = null;
		try {
			FileReader localFileReader = new FileReader(
					"/sys/class/net/wlan0/address");
			BufferedReader localBufferedReader = new BufferedReader(
					localFileReader);
			wifiMacAddress = localBufferedReader.readLine();
			if (wifiMacAddress == null || "null".endsWith(wifiMacAddress)) {
				wifiMacAddress = Constant.unknown;
			}
			if (!Constant.unknown.endsWith(wifiMacAddress)) {
				wifiMacAddress = wifiMacAddress.trim();
				Log.d(TAG, "wifiMacAddress = " + wifiMacAddress);
				wifiMacAddress = wifiMacAddress.replace(":", "-").toUpperCase();
			}
			localBufferedReader.close();
			localFileReader.close();
			setWifiMacAddr(wifiMacAddress.trim());
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	// 获取激活的无线网卡mac
	// private void getWifiMacAddress() {
	// String wifiMacAddress = null;
	// try {
	// WifiManager mWifiManager = (WifiManager) mContext
	// .getSystemService(Context.WIFI_SERVICE);
	// boolean isWifiEnabled = mWifiManager.isWifiEnabled();
	// Log.d(TAG, "isWifiEnabled = " + isWifiEnabled);
	// if (isWifiEnabled) {
	// WifiInfo mWifiInfo = mWifiManager.getConnectionInfo();
	// if (mWifiInfo != null) {
	// wifiMacAddress = mWifiInfo.getMacAddress();
	//
	// }
	// }
	// if (wifiMacAddress == null
	// || "null".equalsIgnoreCase(wifiMacAddress)) {
	// wifiMacAddress = Constant.unknown;
	// }
	// if (!Constant.unknown.endsWith(wifiMacAddress)) {
	// wifiMacAddress = wifiMacAddress.replace(":", "-").toUpperCase();
	// }
	// Log.d(TAG, "wifiMacAddress = " + wifiMacAddress);
	// setWifiMacAddr(wifiMacAddress);
	//
	// } catch (Exception e) {
	// e.printStackTrace();
	// }
	// }

	private void getUserID(Context context) {
		String[] str_user = { "userid", "create_time" };
		String uriString = "content://com.android.tcl.messagebox/message_user";
		Uri uri = Uri.parse(uriString);
		try {
			Cursor c = context.getContentResolver().query(uri, str_user, null,
					null, null);

			if (c != null && c.getCount() > 0) {
				c.moveToFirst();
				userId = c.getString(c.getColumnIndex("userid")) == null ? ""
						: c.getString(c.getColumnIndex("userid")).trim();

			} else {
				userId = "";
			}
			Log.d(TAG, "userId = " + userId);
			if (c != null && !c.isClosed()) {
				c.close();
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String get_IdentifyId() {

		SharedPreferences preferences = mContext.getSharedPreferences(
				Constant.CONFIG_FILE, Context.MODE_PRIVATE);
		String identifyId = preferences.getString(Constant.identify_id, "");
		Log.d(TAG, "identifyId = " + identifyId);
		setIdentifyId(identifyId);
		if (identifyId != null)
			identifyId = identifyId.trim();
		return identifyId;

	}

	private synchronized void getHuanId() {
		ContentResolver resolver = mContext.getContentResolver();

		// ningyb 全球播终端获取huanid的方法
		String columns[] = new String[] { MyUsers.devicetoken.ACTIVE_FLAG,
				MyUsers.devicetoken.DEVICE_ID, MyUsers.devicetoken.DUM,
				MyUsers.devicetoken.DEVICE_MODEL,
				MyUsers.devicetoken.ACTIVE_KEY, MyUsers.devicetoken.DIDTOKEN,
				MyUsers.devicetoken.TOKEN, MyUsers.devicetoken.HUAN_ID,
				MyUsers.devicetoken.LICENSE_TYPE,
				MyUsers.devicetoken.LICENSE_DATA };
		Uri myUri = MyUsers.devicetoken.CONTENT_URI;
		String huanId = null;
		try {
			Cursor cur = resolver.query(myUri, columns, null, null, null);
			huanId = "";
			if (cur != null) {
				if (cur.moveToFirst()) {
					do {
						huanId = cur.getString(cur
								.getColumnIndex(MyUsers.devicetoken.HUAN_ID));
					} while (cur.moveToNext());
				}
				cur.close();
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if (huanId == null) {
			huanId = "";
		}
		Log.d(TAG, "huanId->" + huanId);

		setHuan_id(huanId.trim());
	}

	// appname
	public String getAppRunningTop() {
		ActivityManager activityManager = (ActivityManager) mContext
				.getSystemService(Context.ACTIVITY_SERVICE);
		List<RunningTaskInfo> tasksInfo = activityManager.getRunningTasks(1);
		String appName = null;
		if (tasksInfo.size() > 0) {
			appName = tasksInfo.get(0).topActivity.getPackageName();

		}
		Log.d(TAG, "appName=" + appName);

		return appName.trim();
	}

	private void getTerminalVersion() {

		try {
			PackageInfo packInfo = mContext.getPackageManager().getPackageInfo(
					mContext.getPackageName(), 0);
			setVersionName(packInfo.versionName.trim());

		} catch (NameNotFoundException e) {
			e.printStackTrace();
		}

	}

	private void get_Dnum() {
		
		try {
			SqlCommon mSqlcommon = new SqlCommon();
			ContentResolver mResolver = mContext.getContentResolver();
			Log.d(TAG, "get_Dnum=" + mSqlcommon.getDum(mResolver));
			setDnum(mSqlcommon.getDum(mResolver).trim());
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private String get_PatternlibVersion() {

		SharedPreferences preferences = mContext.getSharedPreferences(
				Constant.CONFIG_FILE, Context.MODE_PRIVATE);
		String patternlibVersion = preferences.getString(
				Constant.PatternLibVersion, "");
		Log.d(TAG, "patternlibVersion = " + patternlibVersion);

		if ((patternlibVersion == null) || (patternlibVersion == "")) {
			patternlibVersion = "1.5.1";
		}
		setPatternlibVersion(patternlibVersion);
		return patternlibVersion.trim();

	}

	// private String[][] get_TimeScaling(){
	// SharedPreferences preferences = mContext.getSharedPreferences(
	// Constant.ScalingTime_FILE, Context.MODE_PRIVATE);
	// // read ScalingTime
	// Set<String> timeSet = new TreeSet<String>();
	//
	// timeSet = preferences.getStringSet("cctv", timeSet);
	// String[] data = (String[]) timeSet.toArray(new String[timeSet
	// .size()]); // 将SET转换为数组
	//
	//
	// //二维数组
	// }

	public String getDeviceId() {
		if (deviceId != null)
			deviceId = deviceId.trim();
		else
			deviceId = "000";
		return deviceId;
	}

	private void setDeviceId(String deviceId) {
		this.deviceId = deviceId;
	}

	public String getDevModel() {
		return devModel;
	}

	private void setDevModel(String devModel) {
		this.devModel = devModel;
	}

	public String getHuan_id() {
		return huan_id;
	}

	private void setHuan_id(String huan_id) {
		this.huan_id = huan_id;
	}

	public String getSv() {
		return sv;
	}

	private void setSv(String sv) {
		this.sv = sv;
	}

	public InetAddress getIp() {
		return ip;
	}

	private void setIp(InetAddress ip) {
		this.ip = ip;
	}

	public String getMacAddr() {
		return macAddr;
	}

	private void setMacAddr(String macAddr) {
		this.macAddr = macAddr;
	}

	private void setUserId(String userId) {
		this.userId = userId;
	}

	public String getUserId() {
		return userId;
	}

	private void setVersionName(String version) {
		this.versionName = version;
	}

	public String getVersionName() {
		return versionName;
	}

	public String getDnum() {
		return dnum;
	}

	private void setDnum(String dnum) {
		this.dnum = dnum;
	}

	public String getWifiMacAddr() {
		return wifiMacAddr;
	}

	private void setWifiMacAddr(String wifiMacAddr) {
		this.wifiMacAddr = wifiMacAddr;
	}

	public String getIdentifyId() {
		return identifyId;
	}

	public void setIdentifyId(String identifyId) {
		this.identifyId = identifyId;
	}

	public String getPatternlibVersion() {
		return patternlibVersion;
	}

	private void setPatternlibVersion(String patternlibVersion) {
		this.patternlibVersion = patternlibVersion;
	}

	public String getApp_name() {
		return app_name;
	}

	public void setApp_name(String app_name) {
		this.app_name = app_name;
	}
}
