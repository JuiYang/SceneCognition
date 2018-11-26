package com.tcl.recognize.util;

public class ErrorCode {
	private static final String UNDEFINED_ERRORCODE = "undefined error code";

	public static class ErrorCode1 {
		// private static final String err_1000 = "success";
		private static final String err_1001 = "input parameter errror";
		private static final String err_1002 = "system error";
		private static final String err_1003 = "black fingerprint";

		public static String getErrorString(int errorCode) {
			String errString = null;
			switch (errorCode) {
			case 1000:
				errString = null;
				break;
			case 1001:
				errString = err_1001;
				break;
			case 1002:
				errString = err_1002;
				break;
			case 1003:
				errString = err_1003;
				break;
			default:
				errString = UNDEFINED_ERRORCODE;
				break;
			}
			return errString;
		}
	}

	public static class ErrorCode2 {
		// private static final String err_0000 = "success";
		private static final String err_1001 = "no records exist";
		private static final String err_1111 = "system error";
		private static final String err_2001 = "input parameter errror";
		private static final String err_2002 = "permission denied";

		public static String getErrorString(int errorCode) {
			String errString = null;
			switch (errorCode) {
			case 0000:
				errString = null;
				break;
			case 1001:
				errString = err_1001;
				break;
			case 1111:
				errString = err_1111;
				break;
			case 2001:
				errString = err_2001;
				break;
			case 2002:
				errString = err_2002;
				break;
			default:
				errString = UNDEFINED_ERRORCODE;
				break;
			}
			return errString;
		}
	}
	
	public static final String errString_screenshot = "截屏失败";
	public static final String errString_screenshot_4k2k = "4k2k片源不支持截屏";
	public static final String errString_search = "请求服务失败";
	public static final String errString_result = "请求服务失败";
	public static final String errString_canceled = "识别取消";
	public static final String errString_netWork = "请检查网络连接";
	
	
	
}
