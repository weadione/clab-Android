package com.example.newpaintlapse

import android.graphics.SurfaceTexture
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.view.ViewGroup
import com.example.newpaintlapse.databinding.ActivityMainBinding
import com.google.mediapipe.components.*
import com.google.mediapipe.formats.proto.LandmarkProto
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.GraphTextureFrame
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import java.util.Timer
import kotlin.concurrent.fixedRateTimer

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val BINARY_GRAPH_NAME = "pose_world_gpu.binarypb"
    private val VIDEO_STREAM_NAME = "input_video"
    private val HAR_INPUT_LANDMARKS = "pose_world_landmarks"
    private val PFD_INPUT_LANDMARKS = "pose_landmarks"
    private val PFD_INPUT_VIDEO = "input_video"
    private val CAMERA_FACING = CameraHelper.CameraFacing.BACK
    private val FLIP_FRAMES_VERTICALLY = true


    private lateinit var binding: ActivityMainBinding


    //mediapipe var
    private var previewFrameTexture: SurfaceTexture? = null
    private var previewDisplayView: SurfaceView? = null
    private var eglManager: EglManager? = null
    private var processor: FrameProcessor? = null
    private var converter: ExternalTextureConverter? = null
    private var cameraHelper: CameraXPreviewHelper? = null

    // helper and util variables
    private lateinit var harHelper: HARHelper

    init {
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        previewDisplayView = SurfaceView(this)

        initHelperClass()
        initMediapipe()
        addPacketCallbacks()
        harHelper.initOrt()
        PermissionHelper.checkAndRequestCameraPermissions(this);
    }

    override fun onResume() {
        super.onResume()
        converter = ExternalTextureConverter(eglManager!!.context)
        converter!!.setFlipY(FLIP_FRAMES_VERTICALLY)
        converter!!.setConsumer(processor)
        if (PermissionHelper.cameraPermissionsGranted(this)) {
            startCamera()
        }

        harHelper.startSkeletonTimer(binding.harLabel)

    }

    override fun onPause() {
        super.onPause()
        converter!!.close()
        harHelper.stopSkeletonTimer()

    }

    fun initHelperClass(){
        harHelper = HARHelper(this )
    }

    fun initMediapipe(){
        setPreviewDisplay()

        AndroidAssetUtil.initializeNativeAssetManager(this)
        eglManager = EglManager(null)
        processor = FrameProcessor(
            this,
            eglManager!!.nativeContext,
            BINARY_GRAPH_NAME,
            VIDEO_STREAM_NAME,
            VIDEO_STREAM_NAME
        )

        processor!!
            .videoSurfaceOutput
            .setFlipY(FLIP_FRAMES_VERTICALLY)
    }

    fun addPacketCallbacks(){
        processor!!.addPacketCallback(
            HAR_INPUT_LANDMARKS
        ){ packet: Packet ->
            val landmarksRaw: ByteArray = PacketGetter.getProtoBytes(packet)
            val poseLandmarks: LandmarkProto.LandmarkList =
                LandmarkProto.LandmarkList.parseFrom(landmarksRaw)
                harHelper.saveSkeletonData(poseLandmarks)
        }

//        processor!!.addPacketCallback(
//            PFD_INPUT_VIDEO
//        ){ packet: Packet ->
//            val videoTextureFrame: GraphTextureFrame = PacketGetter.getTextureFrame(packet)
//            videoTextureFrame.
//
//            harHelper.saveSkeletonData(poseLandmarks)
//        }
    }

    private fun setPreviewDisplay(){
        previewDisplayView!!.visibility= View.GONE
        var view_group: ViewGroup = findViewById(R.id.preview_display_layout)
        view_group.addView(previewDisplayView)

        previewDisplayView!!
            .holder
            .addCallback(
                object : SurfaceHolder.Callback {
                    override fun surfaceCreated(holder: SurfaceHolder) {
                        processor!!.videoSurfaceOutput.setSurface(holder.surface)
                        Log.d("Surface", "Surface Created")
                    }

                    override fun surfaceChanged(
                        holder: SurfaceHolder,
                        format: Int,
                        width: Int,
                        height: Int
                    ) {
                        onPreviewDisplaySurfaceChanged(holder, format, width, height)
                    }

                    override fun surfaceDestroyed(holder: SurfaceHolder) {
                        processor!!.videoSurfaceOutput.setSurface(null)
                    }
                })

    }

    private fun onPreviewDisplaySurfaceChanged(
        holder: SurfaceHolder?, format: Int, width: Int, height: Int
    ) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        val viewSize: Size = Size(width, height)
        val displaySize = cameraHelper!!.computeDisplaySizeFromViewSize(viewSize)
        val isCameraRotated = cameraHelper!!.isCameraRotated

        // Connect the converter to the camera-preview frames as its input (via
        // previewFrameTexture), and configure the output width and height as the computed
        // display size.
        converter!!.setSurfaceTextureAndAttachToGLContext(
            previewFrameTexture,
            if (isCameraRotated) displaySize.height else displaySize.width,
            if (isCameraRotated) displaySize.width else displaySize.height
        )
    }

    private fun startCamera() {
        cameraHelper = CameraXPreviewHelper()
        cameraHelper!!.setOnCameraStartedListener { surfaceTexture: SurfaceTexture? ->
            previewFrameTexture = surfaceTexture
            // Make the display view visible to start showing the preview. This triggers the
            // SurfaceHolder.Callback added to (the holder of) previewDisplayView.
            previewDisplayView!!.visibility = View.VISIBLE
        }
        cameraHelper!!.startCamera(this, CAMERA_FACING,  /*surfaceTexture=*/null)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults)
    }

}