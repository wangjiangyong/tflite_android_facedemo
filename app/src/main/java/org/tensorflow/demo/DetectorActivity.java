/*
 * Copyright 2018 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.demo;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.Camera;
import android.media.ImageReader.OnImageAvailableListener;

import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.KeyEvent;
import android.view.View;
import android.content.Intent;

import android.widget.Toast;
import android.view.View.OnLongClickListener;
import android.view.View.OnClickListener;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;


import org.tensorflow.demo.OverlayView.DrawCallback;
import org.tensorflow.demo.env.BorderedText;
import org.tensorflow.demo.env.ImageUtils;
import org.tensorflow.demo.env.Logger;
import org.tensorflow.demo.tracking.MultiBoxTracker;
import org.tensorflow.lite.demo.R; // Explicit import needed for internal Google builds.

import static android.content.ContentValues.TAG;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  public static final String EXTRA_MESSAGE = "org.tensorflow.demo.MESSAGE";

  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect-class1.tflite";
  private static final String TF_OD_API_LABELS_FILE = "face_labels_list.txt";
  
  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;

  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.7f;
  private static final float MINIMUM_CONFIDENCE_MXNET_FACE_RECOGNITION = 0.4f;  //wjy  人脸识别余弦距离阈值
  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;

  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private long ProEmbFeatureMS;
  private int personNum;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private Bitmap rgbFrameBitmap1 = null;   //wjy

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  private Matrix frameTo1Transform;   //wjy

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;
  private float[] reg_person_embfeature;   //wjy
  private String reg_person_name;   //wjy
  //wjy add face recognition
  public static FaceNetMobile mFace = new FaceNetMobile();
  private static final int REQUEST_EXTERNAL_STORAGE = 1;
  private static String[] PERMISSIONS_STORAGE = {
          "android.permission.READ_EXTERNAL_STORAGE",
          "android.permission.WRITE_EXTERNAL_STORAGE" };


  private void copyBigDataToSD(String strOutFileName) throws IOException {
    Log.i(TAG, "start copy file " + strOutFileName);
    File sdDir = Environment.getExternalStorageDirectory();//get directory
    File file = new File(sdDir.toString()+"/facem/");
    LOGGER.i("wjy debug " + sdDir.toString() + "/facem/" );    //wjy
    if (!file.exists()) {
      file.mkdir();
    }

    String tmpFile = sdDir.toString()+"/facem/" + strOutFileName;
    File f = new File(tmpFile);
    if (f.exists()) {
      Log.i(TAG, "file exists " + strOutFileName);
      return;
    }
    InputStream myInput;
    java.io.OutputStream myOutput = new FileOutputStream(sdDir.toString()+"/facem/"+ strOutFileName);
    myInput = this.getAssets().open(strOutFileName);
    byte[] buffer = new byte[1024];
    int length = myInput.read(buffer);
    while (length > 0) {
      myOutput.write(buffer, 0, length);
      length = myInput.read(buffer);
    }
    myOutput.flush();
    myInput.close();
    myOutput.close();
    Log.i(TAG, "end copy file " + strOutFileName);
  }


  @Override
  protected void onCreate(final Bundle savedInstanceState) {
    LOGGER.d("onCreate " + this);
    super.onCreate(null);

    Intent intent = getIntent();
    reg_person_embfeature = intent.getFloatArrayExtra(EXTRA_MESSAGE);
    reg_person_name = intent.getStringExtra("PersonName");

    try {
      copyBigDataToSD("det1.bin");
      copyBigDataToSD("det2.bin");
      copyBigDataToSD("det3.bin");
      copyBigDataToSD("det1.param");
      copyBigDataToSD("det2.param");
      copyBigDataToSD("det3.param");
      copyBigDataToSD("recognition.bin");
      copyBigDataToSD("recognition.param");
    } catch (IOException e) {
      e.printStackTrace();
    }
    //model init
    File sdDir = Environment.getExternalStorageDirectory();//get directory
    String sdPath = sdDir.toString() + "/facem/";
    mFace.FaceModelInit(sdPath);


  }

  //wjy add
  private class PicOnLongClick implements OnLongClickListener{
    @Override
    public boolean onLongClick(View view){
      try{
        Intent intent = new Intent(DetectorActivity.this,RegPersonActivity.class);
        startActivity(intent);

      }
      catch(Exception e){
          Toast.makeText(getApplicationContext(), "注册界面错误！",Toast.LENGTH_LONG).show();
      }
      return true;
    }
  }

  //wjy add
  private class PicOnShortClick implements OnClickListener{
    @Override
    public void onClick(View view){
      try{
        setDebugMode();
      }
      catch(Exception e){
        Toast.makeText(getApplicationContext(), "Debug模式开启失败！",Toast.LENGTH_LONG).show();
      }

    }
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {   //初始化执行，一次
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;



    } catch (final IOException e) {
      LOGGER.e("Exception initializing classifier!", e);
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }


    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("wjy debug xxx: %d  %d", rotation,getScreenOrientation());
    LOGGER.i("wjy debug xxx sensorOrientation " + sensorOrientation);    //wjy
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888 );


    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    if( LegacyCameraConnectionFragment.camera_id == Camera.CameraInfo.CAMERA_FACING_BACK) {
      cropToFrameTransform = new Matrix();
      frameToCropTransform.invert(cropToFrameTransform);
    }
    else if(LegacyCameraConnectionFragment.camera_id == Camera.CameraInfo.CAMERA_FACING_FRONT){ //考虑前置摄像头
      cropToFrameTransform = new Matrix();
      frameToCropTransform.invert(cropToFrameTransform);
    }

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.setOnLongClickListener(new PicOnLongClick()); //wjy
    trackingOverlay.setOnClickListener(new PicOnShortClick()); //wjy


    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);   //wjy  跟踪调试信息
            }
          }
        });


    addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            if (!isDebug()) {
              return;
            }

            final Bitmap copy = cropCopyBitmap;

            if (copy == null) {
              return;
            }

            final int backgroundColor = Color.argb(100, 0, 0, 0);
            canvas.drawColor(backgroundColor);

            final Matrix matrix = new Matrix();
            final float scaleFactor = 2;
            matrix.postScale(scaleFactor, scaleFactor);
            matrix.postTranslate(
                canvas.getWidth() - copy.getWidth() * scaleFactor,
                canvas.getHeight() - copy.getHeight() * scaleFactor);

            canvas.drawBitmap(copy, matrix, new Paint());       //绘制右下缩略图，标示为检测人脸的效果，不包含跟踪

            final Vector<String> lines = new Vector<String>();
            if (detector != null) {
              final String statString = detector.getStatString();
              final String[] statLines = statString.split("\n");
              for (final String line : statLines) {
                lines.add(line);
              }
            }
            lines.add("");

            lines.add("Frame: " + previewWidth + "x" + previewHeight);
            lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
            lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
            lines.add("Rotation: " + sensorOrientation);
            lines.add("Registered Face: " + reg_person_name );
            lines.add("Detect Number ofFaces: " + personNum );
            lines.add("Face DetInference time: " + lastProcessingTimeMs + "ms");
            lines.add("Face EmbeddingFeature time:" + ProEmbFeatureMS + "ms");

            borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
          }
        });
  }

  OverlayView trackingOverlay;

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = getLuminance();
    tracker.onFrame(
        previewWidth,
        previewHeight,
        getLuminanceStride(),
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();  //界面刷新,postInvalidate 在非UI线程中使用

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    //ImageUtils.saveBitmap(rgbFrameBitmap,"wjy.jpg");



    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();


    final Canvas canvas = new Canvas(croppedBitmap);  //croppedBitmap创建以后，通过canvas.drawBitmap获取变换后的bitmap
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {     // 匿名类
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              //LOGGER.i("wjy debug fuck**" +result.getTitle());    //wjy
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }
            //wjy add
            personNum = mappedRecognitions.size(); //wjy 剪到的人脸数,人脸置信度超过阈值。
            getPersonBitmapListAndRecognition(mappedRecognitions);


            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();   //界面刷新,postInvalidate 在非UI线程中使用

            requestRender();
            computingDetection = false;

          }
        });
  }

  //wjy add
  private void getPersonBitmapListAndRecognition(final List<Classifier.Recognition> mappedRecognitions) {

    //final List<Bitmap> personList = new LinkedList<Bitmap>();
    //wjy
    frameTo1Transform = new Matrix();
    frameTo1Transform.postScale(1f, 1f);
    frameTo1Transform.postRotate(sensorOrientation);

    //wjy add
    if(mappedRecognitions.size()<=0){
      return;
    }
    for(final Classifier.Recognition result : mappedRecognitions) {
        int left1 = (int) result.getLocation().left;
        int right1 = (int) result.getLocation().right;
        int top1 = (int) result.getLocation().top;
        int bottom1 = (int) result.getLocation().bottom;
        LOGGER.i("wjy debug result left" + left1);    //wjy
        LOGGER.i("wjy debug result right" + right1);    //wjy
        LOGGER.i("wjy debug result top" + top1);    //wjy
        LOGGER.i("wjy debug result bottom" + bottom1);    //wjy

        //bug to fix  "java.lang.IllegalArgumentException: x + width must be <= bitmap.width()"  OK
        if(right1>rgbFrameBitmap.getWidth()||bottom1>rgbFrameBitmap.getHeight()||top1<=0||left1<=0||right1<=0||bottom1<=0)  //过滤超出边界的BBox
          return;
        rgbFrameBitmap1 = Bitmap.createBitmap(rgbFrameBitmap, left1,top1, right1-left1,bottom1-top1, frameTo1Transform, true);
        byte[] faceDate1 = RegPersonActivity.getPixelsRGBA(rgbFrameBitmap1);
        final long startTime = SystemClock.uptimeMillis();
        float[] embFeature1 = mFace.FaceEmbFeature(faceDate1,rgbFrameBitmap1.getWidth(),rgbFrameBitmap1.getHeight());
        ProEmbFeatureMS = SystemClock.uptimeMillis() - startTime;

        if(reg_person_embfeature == null||embFeature1==null)
          return;
        double cmpValue =  mFace.calculSimilar(reg_person_embfeature,embFeature1);
        LOGGER.i("wjy debug cmpValue " + cmpValue);
        if (cmpValue > MINIMUM_CONFIDENCE_MXNET_FACE_RECOGNITION){
            result.setTitle(reg_person_name);
            result.setConfidence((float)cmpValue);
        }
        //ImageUtils.saveBitmap(rgbFrameBitmap1, "wjy1.jpg");
    }
  }

  //wjy add   精度问题，弃用，调用Jni计算
  private double calculSimilar(float[] v1,float[] v2) {
    if(v1.length != v2.length||v1.length!=128)
      return 0;
    double ret = 0.0, mod1 = 0.0, mod2 = 0.0;
    for (int i = 0; i != v1.length; ++i)
    {
      ret += v1[i] * v2[i];
      mod1 += v1[i] * v1[i];
      mod2 += v2[i] * v2[i];
    }
    return  ret / mod1*mod1 / mod2*mod2 ;
  }


    @Override   //wjy 切换镜头后修改变换矩阵
    public boolean onKeyDown(final int keyCode, final KeyEvent event) {
        boolean reval = super.onKeyDown(keyCode, event);  //wjy 调用父类事件响应函数
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP) {
            int cropSize = TF_OD_API_INPUT_SIZE;
            if(LegacyCameraConnectionFragment.camera_id == Camera.CameraInfo.CAMERA_FACING_BACK){
                sensorOrientation = 90 - getScreenOrientation();
                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                previewWidth, previewHeight,
                                cropSize, cropSize,
                                sensorOrientation, MAINTAIN_ASPECT);

                cropToFrameTransform = new Matrix();
                frameToCropTransform.invert(cropToFrameTransform);
            }
            else if(LegacyCameraConnectionFragment.camera_id == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                sensorOrientation = 270 - getScreenOrientation();
                frameToCropTransform =
                        ImageUtils.getTransformationMatrix(
                                previewWidth, previewHeight,
                                cropSize, cropSize,
                                sensorOrientation, MAINTAIN_ASPECT);

                cropToFrameTransform = new Matrix();
                frameToCropTransform.invert(cropToFrameTransform);
            }
            tracker.setSensorOrientation(sensorOrientation);//更新跟踪模块的旋转方向
            return true;
        }
        return reval;

    }


  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onSetDebug(final boolean debug) {
    detector.enableStatLogging(debug);
  }



}
