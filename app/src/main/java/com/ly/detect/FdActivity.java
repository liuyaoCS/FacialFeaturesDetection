package com.ly.detect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.video.Video;

import android.R.integer;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.View.OnClickListener;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.SeekBar.OnSeekBarChangeListener;

public class FdActivity extends Activity implements CvCameraViewListener2 {

	private static final String TAG = "OPENCV";
	
	private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);//green
	private static final Scalar EYER_RECT_COLOR = new Scalar(255, 0, 0, 255);//red 
	private static final Scalar EYEL_RECT_COLOR = new Scalar(204, 232, 207, 255);//
	private static final Scalar MOUTH_RECT_COLOR = new Scalar(0, 0, 255, 255);//blue
	private static final Scalar NOSE_RECT_COLOR = new Scalar(255, 255, 0, 255);//yellow
	private static final Scalar RECT_COLORTEST = new Scalar(205, 205, 205, 255);

	private MenuItem mItemFace50;
	private MenuItem mItemFace40;
	private MenuItem mItemFace30;
	private MenuItem mItemFace20;

	private Mat mRgba;
	private Mat mGray;

	private CascadeClassifier mJavaDetector;
	private CascadeClassifier mJavaDetectorEyeR;
	private CascadeClassifier mJavaDetectorEyeL;
	private CascadeClassifier mJavaDetectorMouth;
	private CascadeClassifier mJavaDetectorNose;

	private float mRelativeFaceSize = 0.2f;
	private int mAbsoluteFaceSize = 0;
	private static int EyeScale=5;
	private static int MouthWScale=5;
	private static int MouthHScale=8;
	private static int NoseWScale=7;
	private static int NoseHScale=8;
	private int eyeSize=0;
	private int mouthWSize=0;
	private int mouthHSize=0;
	private int noseWSize=0;
	private int noseHSize=0;
	
	private CameraBridgeViewBase mOpenCvCameraView;
	private Button setting,staticDetect;
	private boolean isCameraBack=true;
	
	ExecutorService eService;
	private int mode=0; //0:normal 1:threadPool
	
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
	private void generateDetectors(){

		mJavaDetector=generateClassifier(R.raw.lbpcascade_frontalface,"lbpcascade_frontalface.xml");
		mJavaDetectorEyeR=generateClassifier(R.raw.haarcascade_righteye_2splits,"haarcascade_eye_right.xml");
		mJavaDetectorEyeL=generateClassifier(R.raw.haarcascade_righteye_2splits,"haarcascade_eye_left.xml");
		mJavaDetectorMouth=generateClassifier(R.raw.haarcascade_mcs_mouth_simple,"haarcascade_mcs_mouth.xml");
		mJavaDetectorNose=generateClassifier(R.raw.haarcascade_mcs_nose_simple,"haarcascade_mcs_nose.xml");
	}
	private void initViews(){
		mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
		//mOpenCvCameraView.setCameraIndex(1);
		mOpenCvCameraView.enableFpsMeter();
		mOpenCvCameraView.enableView();
		
		setting=(Button) findViewById(R.id.setting);
		setting.setOnClickListener(new OnClickListener() {
			
			@Override
			public void onClick(View v) {
				// TODO Auto-generated method stub
				if(isCameraBack){
					mOpenCvCameraView.disableView();
					mOpenCvCameraView.setCameraIndex(1);
					mOpenCvCameraView.enableView();
					isCameraBack=false;
				}else{
					mOpenCvCameraView.disableView();
					mOpenCvCameraView.setCameraIndex(-1);
					mOpenCvCameraView.enableView();
					isCameraBack=true;
				}
				
			}
		});
		
		staticDetect=(Button) findViewById(R.id.staticDetect);
		staticDetect.setOnClickListener(new OnClickListener() {
			
			@Override
			public void onClick(View v) {
				// TODO Auto-generated method stub
				Intent itIntent=new Intent(FdActivity.this,PictureDetectActivity.class);
				startActivity(itIntent);
			}
		});
	}

	@Override
	public void onCreate(Bundle savedInstanceState) {
		Log.i(TAG, "called onCreate");
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.face_detect_surface_view);
		
		eService=Executors.newFixedThreadPool(4);

		initViews();
		
		Log.i(TAG, "OpenCV library load!");
		if (!OpenCVLoader.initDebug()) {
			Log.e(TAG, "OpenCV load not successfully");
		} else {
			//System.loadLibrary("detection_based_tracker");
			Log.i(TAG, "OpenCV load  successfully");
			generateDetectors();
		}
	
	}
	
	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null){
			mOpenCvCameraView.disableView();
		}	
	}

	@Override
	public void onResume() {
		super.onResume();
//		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,
//				mLoaderCallback);
		if(mOpenCvCameraView!=null){
			mOpenCvCameraView.enableView();
		}
	}

	public void onDestroy() {
		super.onDestroy();
		mOpenCvCameraView.disableView();
	}

	public void onCameraViewStarted(int width, int height) {
		mGray = new Mat();
		mRgba = new Mat();
		
	}

	public void onCameraViewStopped() {
		mGray.release();
		mRgba.release();
	}
	Rect eyearea_right;
	Rect eyearea_left;
	Rect mousearea;
	Rect nosearea;
	long time=0;
	@SuppressWarnings("unchecked")
	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

		
		long time=System.currentTimeMillis();
		mRgba = inputFrame.rgba();
		mGray = inputFrame.gray();
		
		
//		Core.flip(mRgba, mRgba, 1);//flip aroud Y-axis
//		Core.flip(mGray, mGray, 1);
		
		
		if (mAbsoluteFaceSize == 0) {
			int height = mGray.rows();
			if (Math.round(height * mRelativeFaceSize) > 0) {
				mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
//				eyeSize=mAbsoluteFaceSize/EyeScale;
//				mouthWSize=mAbsoluteFaceSize/MouthWScale;
//				mouthHSize=mAbsoluteFaceSize/MouthHScale;
//				noseWSize=mAbsoluteFaceSize/NoseWScale;
//				noseHSize=mAbsoluteFaceSize/NoseHScale;
				
				eyeSize=60;
				mouthWSize=60;
				mouthHSize=50;
				noseWSize=60;
				noseHSize=50;
			}
		}
		
		MatOfRect faces = new MatOfRect();
		
		////////////////////////////
		long face_time=System.currentTimeMillis();
		if (mJavaDetector != null){
			mJavaDetector.detectMultiScale(mGray, faces, 1.1, 2,
					Objdetect.CASCADE_SCALE_IMAGE, // 2
					new Size(mAbsoluteFaceSize, mAbsoluteFaceSize),
					new Size());
		}
		Log.i("time", "face:time cast= "+(System.currentTimeMillis()-face_time));
		///////////////////////////////
		
		Rect[] facesArray = faces.toArray();

		for (int i = 0; i < facesArray.length; i++) {

			Rect face = facesArray[i];
			// split it
			
			eyearea_right = new Rect(face.x + face.width / 8,
					(int) (face.y + (face.height / 4.5)),
					(face.width - 2 * face.width / 8) / 2, 
					(int) (face.height / 4));
			eyearea_left = new Rect(face.x + face.width / 8+ (face.width - 2 * face.width / 8) / 2,					
					(int) (face.y + (face.height / 4.5)),
					(face.width - 2 * face.width / 8) / 2, 
					(int) (face.height / 4));

			nosearea=new Rect(face.x+ face.width/3,
					(int) (face.y + (face.height / 2)),
					face.width / 3,
					(int) (face.height / 4));
			
			mousearea=new Rect(face.x+ face.width / 3,
					(int)(face.y+face.height/2+face.height/6),
					face.width/3, 
					(int) (face.height /4));
			
			////////////detect organ//////////////////
			long time_detect=System.currentTimeMillis();
			if(mode==0){				
				Rect eyeRResult=detect_organ(mJavaDetectorEyeR, eyearea_right,eyeSize,eyeSize,EYER_RECT_COLOR,1);		
				Rect eyeLResult=detect_organ(mJavaDetectorEyeL, eyearea_left,eyeSize,eyeSize,EYER_RECT_COLOR,2);
				
				if(eyeRResult==null && eyeLResult==null)return mRgba;			
				
				noseResult=detect_organ(mJavaDetectorNose, nosearea,noseWSize,noseHSize,NOSE_RECT_COLOR,3);
				detect_organ(mJavaDetectorMouth, mousearea,mouthWSize,mouthHSize,MOUTH_RECT_COLOR,4);
			}else{
				Task t_eyeR=new Task(mJavaDetectorEyeR, eyearea_right,eyeSize,eyeSize,EYER_RECT_COLOR,1);
				Task t_eyeL=new Task(mJavaDetectorEyeL, eyearea_left,eyeSize,eyeSize,EYER_RECT_COLOR,2);
				Task t_noseTask=new Task(mJavaDetectorNose, nosearea,noseWSize,noseHSize,NOSE_RECT_COLOR,3);
				Task t_mouthTask=new Task(mJavaDetectorMouth, mousearea,mouthWSize,mouthHSize,MOUTH_RECT_COLOR,4);
				
				FutureTask<Rect> ft1=new FutureTask<Rect>(t_eyeR);
				FutureTask<Rect> ft2=new FutureTask<Rect>(t_eyeL);
				FutureTask<Rect> ft3=new FutureTask<Rect>(t_noseTask);
				FutureTask<Rect> ft4=new FutureTask<Rect>(t_mouthTask);
				
				Future<Rect> f1=(Future<Rect>)eService.submit(ft1);
				Future<Rect> f2=(Future<Rect>)eService.submit(ft2);
				Future<Rect> f3=(Future<Rect>)eService.submit(ft3);
				Future<Rect> f4=(Future<Rect>)eService.submit(ft4);
				
				try {
					f1.get();
					f2.get();
					f3.get();
					f4.get();
					Rect eyeRResult=ft1.get();
					Rect eyeLResult=ft2.get();
					if(eyeRResult==null && eyeLResult==null)return mRgba;	
				} catch (Exception e) {
					// TODO: handle exception
				}
				
			}
			Log.i("time", "time cast= "+(System.currentTimeMillis()-time_detect));
			////////////////////////////////////
			
			Core.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(),FACE_RECT_COLOR, 3);
			
			//Log.i(TAG,"type-"+"0"+" w= "+facesArray[i].width+"h= "+facesArray[i].height);
		}

		faces.release();
		Log.i("time", "frame cast= "+(System.currentTimeMillis()-time));
		return mRgba;
	}
	Rect noseResult;
	private Rect detect_organ(CascadeClassifier clasificator, Rect face,int sizeW,int sizeH,Scalar color,int type) {
		Rect ret=null;
		Mat mROI = mGray.submat(face);
		MatOfRect organRect = new MatOfRect();
		long time_organ=System.currentTimeMillis();
		clasificator.detectMultiScale(mROI, organRect, 1.1, 2,
				Objdetect.CASCADE_FIND_BIGGEST_OBJECT|Objdetect.CASCADE_DO_CANNY_PRUNING|Objdetect.CASCADE_DO_ROUGH_SEARCH, 
				new Size(sizeW, sizeH),new Size());
		Log.i("time", type+":time cast= "+(System.currentTimeMillis()-time_organ));
		Rect[] organArray = organRect.toArray();
		for (int i = 0; i < organArray.length;i++) {
			Rect organ_absolute_rect = organArray[i];
			organ_absolute_rect.x += face.x ;
			organ_absolute_rect.y += face.y ;
			if(type==4 && (noseResult!=null && noseResult.br().y>=organ_absolute_rect.tl().y)){
				Log.i(TAG, "ignore");
			}else{
				Core.rectangle(mRgba, organ_absolute_rect.tl(), organ_absolute_rect.br(),
						color, 2);
			}
			
			Log.i(TAG,"type-"+type+" w= "+organ_absolute_rect.width+"h= "+organ_absolute_rect.height);
			
			ret=organ_absolute_rect;
		}
		
		mROI.release();
		organRect.release();
		
		return ret;
	}
	private Rect detect_organ_pool(CascadeClassifier clasificator, Rect face,int sizeW,int sizeH,Scalar color,int type) {
		Rect ret=null;
		Mat mROI = mGray.submat(face);
		MatOfRect organRect = new MatOfRect();
		long time_organ=System.currentTimeMillis();
		clasificator.detectMultiScale(mROI, organRect, 1.1, 2,
				Objdetect.CASCADE_FIND_BIGGEST_OBJECT|Objdetect.CASCADE_DO_CANNY_PRUNING|Objdetect.CASCADE_DO_ROUGH_SEARCH, 
				new Size(sizeW, sizeH),new Size());
		Log.i("time", type+":time cast= "+(System.currentTimeMillis()-time_organ));
		Rect[] organArray = organRect.toArray();
		for (int i = 0; i < organArray.length;i++) {
			Rect organ_absolute_rect = organArray[i];
			organ_absolute_rect.x += face.x ;
			organ_absolute_rect.y += face.y ;
			if(type==4 && (noseResult!=null && noseResult.br().y>=organ_absolute_rect.tl().y)){
				Log.i(TAG, "ignore");
			}else{
				Core.rectangle(mRgba, organ_absolute_rect.tl(), organ_absolute_rect.br(),
						color, 2);
			}
			
			Log.i(TAG,"type-"+type+" w= "+organ_absolute_rect.width+"h= "+organ_absolute_rect.height);
			
			ret=organ_absolute_rect;
		}
		
		mROI.release();
		organRect.release();
		
		return ret;
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		Log.i(TAG, "called onCreateOptionsMenu");
		mItemFace50 = menu.add("Face size 50%");
		mItemFace40 = menu.add("Face size 40%");
		mItemFace30 = menu.add("Face size 30%");
		mItemFace20 = menu.add("Face size 20%");
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		Log.i(TAG, "called onOptionsItemSelected; selected item: " + item);
		if (item == mItemFace50){
			setMinFaceSize(0.5f);
		}else if (item == mItemFace40){
			setMinFaceSize(0.4f);
		}else if (item == mItemFace30){
			setMinFaceSize(0.3f);
		}else if (item == mItemFace20){
			setMinFaceSize(0.2f);
		}
		return true;
	}
	private void setMinFaceSize(float faceSize) {
		mRelativeFaceSize = faceSize;
		mAbsoluteFaceSize = 0;
	}
	class Task implements Callable<Rect>{
		CascadeClassifier mClasificator;
		Rect mFace;
		int mSizeW;
		int mSizeH;
		Scalar mColor;
		int mType;
		public  Task(CascadeClassifier clasificator, Rect face, int sizeW, int sizeH, Scalar color, int type) {
			// TODO Auto-generated constructor stub
			mClasificator=clasificator;
			mFace=face;
			mSizeW=sizeW;
			mSizeH=sizeH;
			mColor=color;
			mType=type;
		}
		@Override
		public Rect call() throws Exception {
			// TODO Auto-generated method stub
			return detect_organ_pool(mClasificator, mFace, mSizeW, mSizeH, mColor, mType);
		}
	}
}
