package com.tcl.weilong.mace;
import android.graphics.Bitmap;


import com.tcl.recognize.util.Constant;

import java.nio.FloatBuffer;

public class DelInput {
    private int[] colorValues;
    private FloatBuffer floatBuffer;
//    public static final int InputWidth = 256;
//    public static final int InputHeight = 256;
    /**
     * 输入数据处理，将照片转换成数组 float[]，bitmap是256*256的大小
     * @param bitmap 要处理的原始图片
     * @return 将图片处理后转换成像素点进行返回
     */
    public FloatBuffer BitmapConvertFloat(Bitmap bitmap){
        colorValues = new int[Constant.INPUTWIDTH* Constant.INPUTHEIGHT];
        float[] floatValues = new float[Constant.INPUTWIDTH* Constant.INPUTHEIGHT * 3];
        floatBuffer = FloatBuffer.wrap(floatValues, 0, Constant.INPUTWIDTH* Constant.INPUTHEIGHT * 3);
        bitmap.getPixels(colorValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        floatBuffer.rewind();
        for (int i = 0; i < colorValues.length; i++) {
            int value = colorValues[i];
            floatBuffer.put((((value >> 16) & 0xFF) - 128f) / 128f);
            floatBuffer.put((((value >> 8) & 0xFF) - 128f) / 128f);
            floatBuffer.put(((value & 0xFF) - 128f) / 128f);
        }
        return floatBuffer;
    }
}
