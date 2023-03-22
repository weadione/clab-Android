package com.example.mlikitpaintlape

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.OrientationEventListener
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.core.impl.utils.Exif
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.LifecycleCameraController
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import com.example.mlkitpaintlapse.databinding.ActivityMainBinding
import com.example.newpaintlapse.HARHelper
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import org.w3c.dom.Text
import java.util.concurrent.Executor
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


@ExperimentalGetImage private class CameraAnalyzer(
    private val onPoseDetected: (pose: Pose) -> Unit
): ImageAnalysis.Analyzer{

    private val options = PoseDetectorOptions.Builder()
            .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
            .build()

    private val poseDetector = PoseDetection.getClient(options)

    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: return
        val inputImage =
            InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        poseDetector.process(inputImage)
            .addOnSuccessListener{pose ->
                onPoseDetected(pose)
            }
            .addOnFailureListener{e->
                Log.e("CameraAnalyzer","Pose Error:",e)
            }
            .addOnCompleteListener {
                imageProxy.close()
                mediaImage.close()
            }
    }

}

@ExperimentalGetImage class MainActivity : AppCompatActivity() {


    private lateinit var binding: ActivityMainBinding
    private lateinit var captureView: ImageView
    private lateinit var mOrientationEventListener: OrientationEventListener
    private lateinit var harLabel: TextView

    // helper and util variables
    private lateinit var harHelper: HARHelper
    private lateinit var cameraExecutor: ExecutorService

    private var imageCapture: ImageCapture? = null

    private val onPoseDetected: (pose: Pose) -> Unit = { pose ->
        try{
            harHelper.setSize(binding.viewFinder.width, binding.viewFinder.height)
            val allPoseLandmarks = pose.allPoseLandmarks
            harHelper.saveSkeletonData(pose, 0)
            Log.v(TAG,allPoseLandmarks[0].position3D.x.toString()+","+allPoseLandmarks[0].position3D.y.toString()+","+allPoseLandmarks[0].position3D.z.toString() )
        }catch(e:Exception){
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if(allPermissionsGranted()){
            startCamera()
        } else{
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }
        harLabel = binding.harLabel
        cameraExecutor = Executors.newSingleThreadExecutor()
        harHelper = HARHelper(this)
        harHelper.initOrt()
    }

    override fun onResume() {
        super.onResume()
        harHelper.setSize( binding.viewFinder.height, binding.viewFinder.width)
        harHelper.startSkeletonTimer(binding.harLabel)
    }

    override fun onPause() {
        super.onPause()
        harHelper.stopSkeletonTimer()
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }

    private fun startCamera(){
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            val imageAnalyzer = ImageAnalysis.Builder()
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor,CameraAnalyzer(onPoseDetected))
                }

            imageCapture = ImageCapture.Builder()
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()

                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture, imageAnalyzer
                )
            } catch (e: Exception){
                Log.e(TAG, "Use case binding fail", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<String>, grantResults:
        IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this,
                    "Permissions not granted by the user.",
                    Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    companion object {
        private const val TAG = "MainActivity"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}