package com.tcl.recognize.util;

import android.util.Log;

public class Constant {

	public static final String TAG = "Constant";
	
	public static final String MODEL_PATH = "model_v128_1126.tflite";
	public static final String LABEL_PATH = "labels.txt";
	public static final String unknown = "unknown";

//	public static final String userIDfileName = "userID.txt";
//	
	public static final String ConfigParametersFile = "configParameters.txt";
	public static final int INPUTWIDTH = 128;
	public static final int INPUTHEIGHT = 128;
	public static final String SAVE_PATH = "/tmp/SceneDetection";
	public static final String SAVE_FILE = "screenshot.jpg";
	
	public static final String START_SEVVICE_BRO = "com.tcl.showmode.action.START_RECOGNITION";
	public static final String STOP_SEVVICE_BRO = "com.tcl.showmode.action.STOP_RECOGNITION";
	public static final String RECOGNIZE_RESULT_BRO = "com.tcl.recognize.tv.action.OUTPUT";

	// 如果是多媒体这边，则为false， 如果是深圳测试则是true
	public static boolean DEBUG = true;
	// 切换mace/tflite: tflite: true; mace: false
	public static boolean MaceOrTflite = true;
//
//	public static final String fromDir = "lib";
//
//	public static final String systemLibDir = "/system/lib";
//
//	public static final int bow = 11;
//
//	public static final int gouzheng = 12;
//
//	public static final int recognizeCloud = 21;
//
//	public static final String Rt95 = "RT95";
//
//	public static final String Am6c = "AM6C";
//
//	public static final String Mstar = "Mstar";
//
//	public static final String Ms901 = "MS901";
//
//	public static final String Ms801 = "MS801";
//
//	public static final String Mt55 = "MT55";
//
//	public static final String Mt55_3600 = "Mt55_3600";
//
//	public static final String Mt55_3700 = "Mt55_3700";
//
//	public static final String app_name = "app_name";
//
//	public static final String inputsource_type = "inputsource_type";
//
//	public static final String channel_name = "channel_name";
//
//	public static final String client_fingerprints = "client_fingerprints";
//
//	public static final String f22 = "f22";
//
//	public static final String f16 = "f16";
//
//	public static final String create_times = "create_times";
//
//	public static final String user_id = "user_id";
//
	public static final String identify_id = "identify_id";
//
//	public static final String dnum = "dnum";
//
//	public static final String message_id = "message_id";
//
//	public static final String method = "method";
//
//	public static final String name = "name";
//
//	public static final String client_type = "client_type";
//
//	public static final String device_id = "device_id";
//
//	public static final String mac = "mac";
//
//	public static final String w_mac = "w_mac";
//
//	public static final String huan_id = "huan_id";
//
//	public static final String sv = "sv";
//
//	public static final String terminal_version = "terminal_version";
//
//	public static final String unknown = "unknown";
//
//	public static final String cctv = "cctv";
//	
//	public static final String logoName = "logo.jpg";
//
////	public static final String dictFile = "dictionary.yml";
//
////	public static final String rangeFile = "range.rng";
//
////	public static final String modelFile = "tv_logo_classifier.model";
//
////	public static final String channellistFile = "channel_list.lst";
//
////	public static final String channelMapping = "channels.json";
//
	public static final String CONFIG_FILE = "config_file";
//
//	public static final String ScalingTime_FILE = "scalingTime_file";
//
//	public static final String ScalingTime = "scalingTime";
//
//	public static final String nowTime = "_nowTime";
//
//	public static final String timeDiff = "_timeDiff";
//
//	public static final String UploadLogoTime = "uploadLogoTime";
//
//	public static final String DownLoadBowLibrary = "DownLoadBowLibrary";
//
	public static final String PatternLibVersion = "patternLibVersion";
//	
//	public static final String HUAN_STRONG_TV = "huan.tv.strongtv";
//	
//	public static final String GZ_ADS = "com.gzads.tvac";
//	
//	public static final String LAUNCHER_GZ_ADS = "com.gzads.tvac.TVACService";
//	
//	public static final String LAUNCHER_HUAN_STRONG_TV = "";  //TODO
//	
//	//每次识别都会发换台通知
//	public static final String SNDGZACTION = "android.intent.action.tclrecognizeresult";
//	//本次识别与上次识别结果不同发换台通知
//	public static final String CHANNEL_SWITCH_ACTION = "com.tcl.recognize.channelSwitch";
//	
//	public static final String PRECHANNEL = "prechannel";
//	
//	public static final String CURCHANNEL = "curchannel";
//	
//	public static  Long reportTime = null;
//	
//	//fanfan
	public static boolean isDebug = true;
////	public static String CHANNELFORFILE = "otherRT95";
	
	public static void debugPrint(String Tag, String content) {
		
		if(isDebug && content != null) {
			Log.v( Tag , content );
		}
	}
}
