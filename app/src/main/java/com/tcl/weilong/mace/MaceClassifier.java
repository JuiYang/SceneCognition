package com.tcl.weilong.mace;


import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;

import com.tcl.recognize.util.Constant;
import com.tcl.recognize.util.LogManagerUtil;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import static com.tcl.recognize.tslite.ImageClassifier.renameFile;

public class MaceClassifier {
    private FloatBuffer input;
    private List<String> labelList = new ArrayList<>();
    private DelInput delInput = new DelInput();
    private DelOutput delOutput = new DelOutput();
    private static final String TAG = MaceClassifier.class.getSimpleName();


    public String classifyFrame(Context context, Bitmap croppedBitmap, String imagePath, long starttime){
        Log.d(TAG, "recognizeImage start");
        long startTime0 = SystemClock.uptimeMillis();
        LogManagerUtil.d(TAG, "imagePath is = " + imagePath);

        // 图片尺度变化
        Bitmap bitmap = Bitmap.createScaledBitmap(croppedBitmap,
                Constant.INPUTWIDTH, Constant.INPUTHEIGHT, false);


        input = delInput.BitmapConvertFloat(bitmap);

        try {
            labelList = delOutput.loadLabelList(context);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Log.i("wwww", "classifyFrame: output");
        // 模型初始化
        AppModel appModel = new AppModel();
        Log.i("wwww", "classifyFrame: appModel");
        appModel.maceMobilenetSetAttrs();
        Log.i("wwww", "classifyFrame: maceMobilenetSetAttr");
        appModel.maceMobilenetCreateEngine();


        Log.d(TAG, "recognizeImage start1");

        long startTime = SystemClock.uptimeMillis();

        // run
        Log.i("wwww", "classifyFrame: run over");
        float[] labelProbArrayFloat;
//        labelProbArrayFloat = appModel.maceMobileTest(input);
        labelProbArrayFloat = appModel.maceMobilenetClassify(input);
        long endTime = SystemClock.uptimeMillis();
        // 返回结果
        String resultText = delOutput.printTopKLabels(labelList,labelProbArrayFloat);

        Log.i("wwww", "onCreate: resultText: "+resultText);

        String costTime = Long.toString(endTime - starttime);
        LogManagerUtil.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        // // ningyb 20180622 Smooth the results across frames.
        if (!Constant.DEBUG) {
            sendBroadCast(context, resultText, costTime);
        }

        // whj 2018-09-05
        if (Constant.DEBUG) {
            String recognizeResult = resultText.substring(0, resultText.indexOf(":"));
            Log.d(TAG, "recognizeResult ====== " + recognizeResult);
            Log.d(TAG, "imagePath ====== " + imagePath);
            String newImagePath = imagePath.substring(0, imagePath.lastIndexOf("."));
            Log.d(TAG, "newImagePath ====== " + newImagePath);
            renameFile(imagePath, newImagePath + "_" + recognizeResult + ".jpg");
        }

        long endTime0 = SystemClock.uptimeMillis();
        String timeRunTs = Long.toString(endTime0 - startTime0);
        Log.d(TAG, "Timecost to run ts is : " + timeRunTs);
        resultText = resultText + "\n" + timeRunTs + "ms/";

        Log.d(TAG, "recognizeImage end");
        croppedBitmap.recycle();

        return resultText;
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

    /**

    public String imageAbsPath = "";
    private Button chosePic ;
    private TextView textPicView;
    private final int CHOOSE_PHOTO = 1;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ImageView iv = findViewById(R.id.iv);
        TextView tv =  findViewById(R.id.tv);
//        chosePic = (Button) findViewById(R.id.picChose);
//        textPicView = (TextView) findViewById(R.id.picText);
//        chosePic.setOnClickListener(new View.OnClickListener() {
//            @Override
//            public void onClick(View v) {
//                if (ContextCompat.checkSelfPermission(MaceClassifier.this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
//                        != PackageManager.PERMISSION_GRANTED){
//                    ActivityCompat.requestPermissions(MaceClassifier.this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE},
//                            1);
//                }else{
//                    openAlum();
//                }
//            }
//        });

        FloatBuffer input;
        List<String> labelList = new ArrayList<>();
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inScaled = false;
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(),R.drawable.snow,options);
        iv.setImageBitmap(bitmap);

        DelInput delInput = new DelInput();
        input = delInput.BitmapConvertFloat(bitmap);
        Context context = getApplicationContext();
        DelOutput delOutput = new DelOutput();
        try {
            labelList = delOutput.loadLabelList(context);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Log.i("wwww", "classifyFrame: output");
        // 模型初始化
        AppModel appModel = new AppModel();
//        Log.i("wwww", "classifyFrame: appModel");
//        appModel.maceMobilenetSetAttrs();
//        Log.i("wwww", "classifyFrame: maceMobilenetSetAttr");
//        appModel.maceMobilenetCreateEngine();

        Log.i("wwww", "classifyFrame: run over");
        float[] labelProbArrayFloat;
        File file = new File(Environment.getExternalStorageDirectory(),
                "timg.jpg");
        String filePath = "/storage/emulated/0/DCIM/Camera/timg.jpg";
//        labelProbArrayFloat = appModel.maceMobilenetClassify(input);
        labelProbArrayFloat = appModel.maceMobileTest(file.getPath());
        String resultText = delOutput.printTopKLabels(labelList,labelProbArrayFloat);
        Log.i("wwww", "onCreate: resultText: "+resultText);
        tv.setText(resultText);    //展示结果
    }

    private void openAlum(){
        Intent intent = new Intent("android.intent.action.GET_INTENT");
        intent.setType("image/*");
        startActivityForResult(intent, CHOOSE_PHOTO);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,  String[] permissions, int[] grantResults) {
        switch(requestCode){
            case 1:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    openAlum();
                }
                break;
                default:break;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        switch(requestCode){
            case CHOOSE_PHOTO:
                break;
                default:
                    break;
        }
    }

    */
}
