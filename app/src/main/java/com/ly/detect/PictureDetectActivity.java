package com.ly.detect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.BitmapFactory.Options;
import android.hardware.Camera.FaceDetectionListener;
import android.os.Bundle;
import android.provider.MediaStore.Images.ImageColumns;
import android.util.Log;
import android.view.Menu;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

public class PictureDetectActivity extends Activity {  
	  
    final private static String TAG = "StaticDetect";  
    final private int PICTURE_CHOOSE = 1;  
    final private static float scale=0f;
  
    private ImageView imageView = null;  
    private Bitmap img = null;  
    private Button detect = null;
    private  Button get=null;
    private TextView result = null;  
    
    private CascadeClassifier faceDetector;
	private CascadeClassifier eyeDetector;
	private CascadeClassifier mouthDetector;
	private CascadeClassifier noseDetector;
	
	private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);//green
	private static final Scalar EYE_RECT_COLOR = new Scalar(255, 0, 0, 255);//red 
	private static final Scalar MOUTH_RECT_COLOR = new Scalar(0, 0, 255, 255);//blue
	private static final Scalar NOSE_RECT_COLOR = new Scalar(255, 255, 0, 255);//yellow
	private static final Scalar RECT_COLORTEST = new Scalar(205, 205, 205, 255);

	Mat imgMat=null;
    @Override  
    protected void onCreate(Bundle savedInstanceState) {  
        super.onCreate(savedInstanceState);  
        setContentView(R.layout.activity_staticdetection);  
        
        initView();
        
        Log.i(TAG, "OpenCV library load!");
		if (!OpenCVLoader.initDebug()) {
			Log.e(TAG, "OpenCV load not successfully");
		} else {
			//System.loadLibrary("detection_based_tracker");
			Log.i(TAG, "OpenCV load  successfully");
			generateDetectors();
		}
        
    }  
    private void initView(){
    	get = (Button) this.findViewById(R.id.get);  
        get.setOnClickListener(new OnClickListener() {  
  
            public void onClick(View arg0) {  
                // get a picture form your phone  
                Intent photoPickerIntent = new Intent(Intent.ACTION_PICK);  
                photoPickerIntent.setType("image/*");  
                startActivityForResult(photoPickerIntent, PICTURE_CHOOSE);  
            }  
        });  
  
        result = (TextView) this.findViewById(R.id.result);  
  
        detect = (Button) this.findViewById(R.id.detect);  
        detect.setVisibility(View.INVISIBLE);  
        detect.setOnClickListener(new OnClickListener() {  
            public void onClick(View arg0) {  
  
            	result.setText("Waiting ...");  
                 
 
                imgMat = new Mat();  
                Utils.bitmapToMat(img, imgMat);  
  
                MatOfRect faceDetections = new MatOfRect();  
                faceDetector.detectMultiScale(imgMat, faceDetections, 1.1, 5,
        				Objdetect.CASCADE_FIND_BIGGEST_OBJECT|Objdetect.CASCADE_DO_CANNY_PRUNING|Objdetect.CASCADE_SCALE_IMAGE, // 2
        				new Size(),new Size());  
   
                int facenum = 0;  
                // Draw a bounding box around each face.  
                for (Rect faceRect : faceDetections.toArray()) {  
                    Core.rectangle(imgMat,new Point(faceRect.x, faceRect.y),new Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height),FACE_RECT_COLOR);     
                    
                    Rect eyearea_right = new Rect(faceRect.x + faceRect.width / 8,
        					(int) (faceRect.y + (faceRect.height / 4.5)),
        					(faceRect.width - 2 * faceRect.width / 8) / 2, 
        					(int) (faceRect.height / 3));
                    
                    Rect eyearea_left = new Rect(faceRect.x + faceRect.width / 8+ (faceRect.width - 2 * faceRect.width / 8) / 2,					
        					(int) (faceRect.y + (faceRect.height / 4.5)),
        					(faceRect.width - 2 * faceRect.width / 8) / 2, 
        					(int) (faceRect.height / 3));

                    Rect nosearea=new Rect(faceRect.x+ faceRect.width/4,
        					(int) (faceRect.y + (faceRect.height / 2)),
        					faceRect.width / 2,
        					(int) (faceRect.height / 3));
        			
                    Rect mousearea=new Rect(faceRect.x+ faceRect.width / 4,
        					(int)(faceRect.y+faceRect.height/2+faceRect.height/6),
        					faceRect.width/2, 
        					(int) (faceRect.height /3));
                    
//                    Core.rectangle(imgMat,eyearea_right.tl(),eyearea_right.br(),RECT_COLORTEST,1);
//                    Core.rectangle(imgMat,eyearea_left.tl(),eyearea_left.br(),RECT_COLORTEST,1);
//                    Core.rectangle(imgMat,nosearea.tl(),nosearea.br(),RECT_COLORTEST,1);
//                    Core.rectangle(imgMat,mousearea.tl(),mousearea.br(),RECT_COLORTEST,1);
                    
                    detect_organ(eyeDetector, eyearea_right,EYE_RECT_COLOR);		
    				detect_organ(eyeDetector, eyearea_left,EYE_RECT_COLOR);
    				detect_organ(noseDetector, nosearea,NOSE_RECT_COLOR);
    				detect_organ(mouthDetector,mousearea,MOUTH_RECT_COLOR);
    				
                    ++facenum;  
                }  
  
                Utils.matToBitmap(imgMat, img);  
                imageView.setImageBitmap(img);  
                result.setText("Facecount: " + facenum);  
  
            }  
        });  
  
        imageView = (ImageView) this.findViewById(R.id.imageView);  
        imageView.setImageBitmap(img);  
    }
    private void detect_organ(CascadeClassifier clasificator, Rect rect,Scalar color) {

		Mat mROI = imgMat.submat(rect);
		MatOfRect organRect = new MatOfRect();

		clasificator.detectMultiScale(mROI, organRect, 1.1, 3,
				Objdetect.CASCADE_SCALE_IMAGE|Objdetect.CASCADE_DO_CANNY_PRUNING, // 2
				new Size(0,0),new Size());
				
		
		Rect[] organArray = organRect.toArray();
		for (int i = 0; i < organArray.length;i++) {
			Rect organ_absolute_rect = organArray[i];
			organ_absolute_rect.x += rect.x ;
			organ_absolute_rect.y += rect.y ;
			
			Core.rectangle(imgMat, organ_absolute_rect.tl(), organ_absolute_rect.br(),
						color, 1);

		}
		
		mROI.release();
		organRect.release();

	}
    private void generateDetectors(){

		faceDetector=generateClassifier(R.raw.haarcascade_frontalface_alt,"lbpcascade_frontalface.xml");
		eyeDetector=generateClassifier(R.raw.haarcascade_mcs_lefteye,"haarcascade_eye_right.xml");
		mouthDetector=generateClassifier(R.raw.haarcascade_mcs_mouth,"haarcascade_mcs_mouth.xml");
		noseDetector=generateClassifier(R.raw.haarcascade_mcs_nose,"haarcascade_mcs_nose.xml");
	}
    private CascadeClassifier generateClassifier(int resId,String fileName){
		CascadeClassifier mDetector = null;
		try{
			InputStream is = getResources().openRawResource(resId);					
			File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
			File CascadeFile = new File(cascadeDir,fileName);					
			FileOutputStream os = new FileOutputStream(CascadeFile);
	
			byte[] buffer = new byte[4096];
			int bytesRead;
			while ((bytesRead = is.read(buffer)) != -1) {
				os.write(buffer, 0, bytesRead);
			}
			is.close();
			os.close();
			
			mDetector = new CascadeClassifier(CascadeFile.getAbsolutePath());					
			if (mDetector.empty()) {
				Log.e(TAG, "Failed to load cascade classifier");
				mDetector = null;
			} else{
				Log.i(TAG, "Loaded cascade classifier from "+ CascadeFile.getAbsolutePath());
						
			}
			
			CascadeFile.delete();
			cascadeDir.delete();
		}catch(Exception e){
			e.printStackTrace();
		}
		return mDetector;
	}
    @Override  
    protected void onActivityResult(int requestCode, int resultCode,  
            Intent intent) {  
        super.onActivityResult(requestCode, resultCode, intent);  
  
        // the image picker callback  
        if (requestCode == PICTURE_CHOOSE) {  
            if (intent != null) {  
  
                Cursor cursor = getContentResolver().query(intent.getData(),  
                        null, null, null, null);  
                cursor.moveToFirst();  
                int idx = cursor.getColumnIndex(ImageColumns.DATA);  
                String fileSrc = cursor.getString(idx);  
  
                Options options = new Options();  
                options.inJustDecodeBounds = true;  
                img = BitmapFactory.decodeFile(fileSrc, options);  
  
                options.inSampleSize = Math.max(1, (int) Math.ceil(Math.max(  
                        (double) options.outWidth / 1024f,  
                        (double) options.outHeight / 1024f)));  
                options.inJustDecodeBounds = false;  
                img = BitmapFactory.decodeFile(fileSrc, options);  
                result.setText("Clik Detect. ==>");  
  
                imageView.setImageBitmap(img);  
                detect.setVisibility(View.VISIBLE);  
            } else {  
                Log.d(TAG, "idButSelPic Photopicker canceled");  
            }  
        }  
    } 
    class Task implements Runnable{

		@Override
		public void run() {
			// TODO Auto-generated method stub
			
		}
    	
    }
  
}  