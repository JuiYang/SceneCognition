package com.tcl.recognize.util;

import java.text.SimpleDateFormat;

public class TimestampUtil {
	private long timestamp=System.currentTimeMillis();
	
	private static SimpleDateFormat df = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss:SSS");
	
	public String getTimeElapsed(){
		long tmp=timestamp;
		timestamp=System.currentTimeMillis();
		return "***"+Long.toString(timestamp-tmp)+"ms***";
	}
	
	public static String getCurrenTimer() {
		//String formattedDate = df.format(c.getTime());
		
		String formattedDate = df.format(System.currentTimeMillis());
		
		return formattedDate;
	}
}
