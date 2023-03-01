package com.example.testing

import android.content.res.AssetManager
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.*
import androidx.appcompat.app.AppCompatActivity
import com.google.mediapipe.components.CameraHelper.CameraFacing
import com.google.mediapipe.components.CameraXPreviewHelper
import com.google.mediapipe.components.ExternalTextureConverter
import com.google.mediapipe.components.FrameProcessor
import com.google.mediapipe.components.PermissionHelper
import com.google.mediapipe.formats.proto.LandmarkProto.*
import com.google.mediapipe.framework.AndroidAssetUtil
import com.google.mediapipe.framework.Packet
import com.google.mediapipe.framework.PacketGetter
import com.google.mediapipe.glutil.EglManager
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.api.math.exp
import org.jetbrains.kotlinx.multik.ndarray.complex.complexDoubleArrayOf
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.append
import org.jetbrains.kotlinx.multik.ndarray.operations.expandDims
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.stack
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val BINARY_GRAPH_NAME = "pose_world_gpu.binarypb"
    private val INPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_LANDMARKS_STREAM_NAME = "pose_world_landmarks"
    private val CAMERA_FACING = CameraFacing.BACK
    private val FLIP_FRAMES_VERTICALLY = true
    private var frame_cnt =1

    //skeleton data for 144frames
    //If the number of frames obtained is less than 144, empty frames are just 0.0f
    private var frames_skeleton = mk.d3array(144,25,3){0.0f}
    companion object {
        init {
            // Load all native libraries needed by the app.
            System.loadLibrary("mediapipe_jni")
            System.loadLibrary("opencv_java3")
        }
    }

    // {@link SurfaceTexture} where the camera-preview frames can be accessed.
    private var previewFrameTexture: SurfaceTexture? = null
    // {@link SurfaceView} that displays the camera-preview frames processed by a MediaPipe graph.
    private var previewDisplayView: SurfaceView? = null
    // Creates and manages an {@link EGLContext}.
    private var eglManager: EglManager? = null
    // Sends camera-preview frames into a MediaPipe graph for processing, and displays the processed
    // frames onto a {@link Surface}.
    private var processor: FrameProcessor? = null
    // Converts the GL_TEXTURE_EXTERNAL_OES texture from Android camera into a regular texture to be
    // consumed by {@link FrameProcessor} and the underlying MediaPipe graph.
    private var converter: ExternalTextureConverter? = null
    // Handles camera access via the {@link CameraX} Jetpack support library.
    private var cameraHelper: CameraXPreviewHelper? = null

    init {
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewDisplayView = SurfaceView(this)
        setPreviewDisplay()

        AndroidAssetUtil.initializeNativeAssetManager(this);
        eglManager = EglManager(null);
        processor = FrameProcessor(
            this,
            eglManager!!.nativeContext,
            BINARY_GRAPH_NAME,
            INPUT_VIDEO_STREAM_NAME,
            OUTPUT_VIDEO_STREAM_NAME
        )
        processor!!
            .videoSurfaceOutput
            .setFlipY(FLIP_FRAMES_VERTICALLY)

        processor!!.addPacketCallback(
            OUTPUT_LANDMARKS_STREAM_NAME
        ){ packet: Packet ->
            val landmarksRaw: ByteArray = PacketGetter.getProtoBytes(packet)
            val poseLandmarks: LandmarkList =
                LandmarkList.parseFrom(landmarksRaw)

            //debug code
            Log.v(TAG, getPoseLandmarksDebugString(poseLandmarks))

            if(frame_cnt<60){//Collect Skeleton data for 60 frames or 4 seconds(4 seconds not yet implemented).
                saveSkeletonData(poseLandmarks) //save landmarks in array shape (144,25,3)
                frame_cnt++
            }
            else{
                saveSkeletonData(poseLandmarks)

                convertSkeletonData()   //todo: convert landmarks to joint, velocity,bone data
                frames_skeleton = mk.d3array(144,25,3){0.0f} // reinitialize landmarks array
                frame_cnt=1

            }
        }


//        har_test()  //har test code
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
    }

    override fun onPause() {
        super.onPause()
        converter!!.close()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        PermissionHelper.onRequestPermissionsResult(requestCode, permissions, grantResults)
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


    protected fun onPreviewDisplaySurfaceChanged(
        holder: SurfaceHolder?, format: Int, width: Int, height: Int
    ) {
        // (Re-)Compute the ideal size of the camera-preview display (the area that the
        // camera-preview frames get rendered onto, potentially with scaling and rotation)
        // based on the size of the SurfaceView that contains the display.
        val viewSize: Size = Size(width, height)
        val displaySize = cameraHelper!!.computeDisplaySizeFromViewSize(viewSize)
        val isCameraRotated = cameraHelper!!.isCameraRotated

        //displaySize.getHeight(); 핸드폰 디스플레이 사이즈를 의미
        //displaySize.getWidth();


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

    //harmodel test code
    private fun har_test(){
        val am = resources.assets
        val harModel =  getModelByteBuffer(am, "HAR_model.tflite")
        val interpreter = Interpreter(harModel)
        Log.d("OUTPUT: ", "11")

        //get random input data buffer
        val inputShape = interpreter.getInputTensor(0).shape()
        val inputBuffer = TensorBuffer.createFixedSize(inputShape, interpreter.getInputTensor(0).dataType())
        val inputData = FloatArray(1 * 3 * 6 * 144 * 25 * 2){ Math.random().toFloat() }
        inputBuffer.loadArray(inputData)
        Log.d("OUTPUT: ", "Input shape: 22}")


        //get output data buffer
        val outputShape = interpreter.getOutputTensor(0).shape()
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, interpreter.getOutputTensor(0).dataType())
        Log.d("OUTPUT: ", "Input shape: 33")

        //run
        interpreter.run(inputBuffer.buffer, outputBuffer.buffer.rewind())

        //print estimated values for 19 labels
        Log.d("OUTPUT0: ", "output value: ${outputBuffer.floatArray[0]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[1]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[2]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[3]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[4]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[5]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[6]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[7]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[8]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[9]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[10]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[11]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[12]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[13]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[14]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[15]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[16]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[17]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[18]}")
        Log.d("OUTPUT1: ", "output value: ${outputBuffer.floatArray[19]}")
    }

    //load model
    @Throws(IOException::class)
    private fun getModelByteBuffer(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }


    //save skeleton data in one frame and append to (144,25,3) ndarray
    private fun saveSkeletonData(poseLandmarks: LandmarkList){
        var one_frame_skeleton = mk.d2array(25,3){0.0f}
        var landmark = poseLandmarks.landmarkList
        for(i: Int in 0..24) {
            if (landmark[i].visibility >= 0.75) {
                one_frame_skeleton[i, 0] = landmark[i].x
                one_frame_skeleton[i, 1] = landmark[i].y
                one_frame_skeleton[i, 2] = landmark[i].z
            }
        }
        frames_skeleton[frame_cnt] = one_frame_skeleton
    }

    //todo: output of this functions is same as model input
    private fun convertSkeletonData(){
        //dummy humman skeleton data to align the input dimension of the model
        val dummy_human_skeleton = mk.d3array(144,25,3) {0.0f}
        val humans_skeleton = mk.stack(frames_skeleton,dummy_human_skeleton)

        //Transpose for processing in multiInput
        val transpose_skeleton = humans_skeleton.transpose(3,1,2,0)

        val input_data = mutiInput(transpose_skeleton)
    }

    //todo: same function as multi_input() in Runner.py
    private fun mutiInput(transposeSkeleton: NDArray<Float, D4>):Any {
        val C = 3
        val T = 144
        val V = 25
        val M = 2

        val joint_tmp = mk.d4array(C,T,V,M){0.0f}
        val velocity = mk.d4array(C*2,T,V,M){0.0f}
        val bone = mk.d4array(C*2,T,V,M){0.0f}
        val joint = mk.stack(transposeSkeleton,joint_tmp)

        for(i in 0 until V){
            val tmp = transposeSkeleton[0 until C,0 until T, i, 0 until M] - transposeSkeleton[0 until C,0 until T, 1, 0 until M]
//            joint.set(i,tmp)
//            joint_tmp.cat(tmp, axis = 2)
        }
        return transposeSkeleton //temp return for avoid error
    }


    //debug code
    private fun getPoseLandmarksDebugString(poseLandmarks: LandmarkList): String {
        val poseLandmarkStr = """
                Pose landmarks: ${poseLandmarks.landmarkCount}
                
                """.trimIndent()

        var poseMarkers = poseLandmarks.landmarkList[16]
        Log.v(
            TAG, """
     ======Degree Of Position]======
     test :${poseMarkers.x},${poseMarkers.y},${poseMarkers.z}

     """.trimIndent()
        )
        return poseLandmarkStr
    }
}