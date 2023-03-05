package com.example.testing

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.graphics.SurfaceTexture
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.*
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.testing.databinding.ActivityMainBinding
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
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.nio.FloatBuffer
import java.util.*
import kotlin.collections.ArrayList
import kotlin.concurrent.fixedRateTimer
import kotlin.math.exp


class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"
    private val BINARY_GRAPH_NAME = "pose_world_gpu.binarypb"
    private val INPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_VIDEO_STREAM_NAME = "input_video"
    private val OUTPUT_LANDMARKS_STREAM_NAME = "pose_world_landmarks"
    private val CAMERA_FACING = CameraFacing.BACK
    private val FLIP_FRAMES_VERTICALLY = true
    private var frame_cnt =1
    private lateinit var binding: ActivityMainBinding

    //skeleton data for 144frames
    //If the number of frames obtained is less than 144, empty frames are just 0.0f
    private var frames_skeleton = mk.d3array(144,25,3){0.0f}
    private var skeletonBuffer =  ArrayList<D2Array<Float>>()


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
    private lateinit var harLabel: TextView
    private var prevSamplingTime: Long = 0
    private var isGraphRunning: Boolean = false


    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null
    init {
        System.loadLibrary("mediapipe_jni");
        System.loadLibrary("opencv_java3");
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL)
        ortSession = CreateOrtSession()


        previewDisplayView = SurfaceView(this)

        harLabel = binding.harLabel
        harLabel.visibility = View.VISIBLE
        setPreviewDisplay()
        val skeletonTimer = fixedRateTimer(name="SkeletonTimer", initialDelay = 0L, period = 4000L){
            getSkelton()
        }

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

            saveSkeletonData(poseLandmarks)
            isGraphRunning = true
        }
        PermissionHelper.checkAndRequestCameraPermissions(this);
    }
    private fun getSkelton() {
        if(isGraphRunning) {
            sampleSkeletonData()
            val inputData = convertSkeletonData()
            val resString = harInference(inputData)
            clearSkeletonData()
            Log.v("label:", resString)
            Log.v("time:", (SystemClock.uptimeMillis() - prevSamplingTime).toString())
            prevSamplingTime = SystemClock.uptimeMillis()
        }
    }
    fun inferenceOrt(inputData: MultiArray<Float, DN>): FloatArray{
        val inputName = ortSession?.inputNames?.iterator()?.next()
        val shape = longArrayOf(1, 3, 144, 25, 2)
        val ortEnv = OrtEnvironment.getEnvironment()
        val floatArrayData = inputData.toFloatArray()
        val buffer =FloatBuffer.allocate(floatArrayData.size)
        buffer.put(floatArrayData)
        buffer.flip()
        ortEnv.use {
            // Create input tensor
            val input_tensor = OnnxTensor.createTensor(ortEnv, buffer, shape)
            input_tensor.use {
                // Run the inference and get the output tensor
                val output = ortSession?.run(Collections.singletonMap(inputName, input_tensor))
                val prediction: OnnxTensor = output?.toList()?.get(0)?.toPair()?.second as OnnxTensor

                val predictionArray: FloatArray = FloatArray(prediction.floatBuffer.remaining())
                prediction.floatBuffer.get(predictionArray)
                return predictionArray
            }
        }
    }
    private fun softmax(input: FloatArray): FloatArray {
        val output = FloatArray(input.size)
        var sum = 0.0f

        // Compute exponential of each element and sum them up
        for (i in input.indices) {
            output[i] = exp(input[i].toDouble()).toFloat()
            sum += output[i]
        }

        // Normalize by dividing each element by the sum
        for (i in output.indices) {
            output[i] /= sum
        }

        return output
    }

    private fun CreateOrtSession(): OrtSession? {
        val so = OrtSession.SessionOptions()
        return ortEnv?.createSession(readModel(), so)
    }

    private fun readModel(): ByteArray {
        return  resources.openRawResource(R.raw.har_gcn).readBytes()
    }
    private fun getLabel(output: FloatArray): String{
        return try {
            val big3 = arrayOf(
                output.sliceArray(0 until 17).sum(),
                output[17],
                output[18]
            )

            var top1Index = 0
            var topValue = big3[0]

            for (i in 1 until big3.size) {
                if (big3[i] > topValue) {
                    top1Index = i
                    topValue = big3[i]
                }
            }

            when (top1Index) {
                0 -> "Other"
                1 -> "Painting:" + String.format("%.1f", (big3[top1Index] * 100)) + "%"
                else -> "Interview:" + String.format("%.1f", (big3[top1Index] * 100)) + "%"
            }
        } catch (e: Exception) {
            e.message?.let { Log.v("HAR", it) }
            ""
        }
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
        skeletonBuffer.add(one_frame_skeleton)
    }

    private fun sampleSkeletonData(){
        val curskeletonBuffer = ArrayList<D2Array<Float>>()
        curskeletonBuffer.addAll(skeletonBuffer)
        skeletonBuffer.clear()

        val frameNum = curskeletonBuffer.size
        Log.v("num",frameNum.toString())
        if(frameNum>=60) {
            val skipInterval = frameNum / 60.0
            for (i in 0 until 60)
                frames_skeleton[i] = curskeletonBuffer[(i * skipInterval).toInt()]
        }
        else
            for(i in 0 until frameNum)
                frames_skeleton[i] = curskeletonBuffer[i]
    }

    //todo: output of this functions is same as model input
    private fun convertSkeletonData() : MultiArray<Float, DN>{
        //dummy humman skeleton data to align the input dimension of the model
        val dummy_human_skeleton = mk.d3array(144,25,3) {0.0f}
        val humans_skeleton = mk.stack(frames_skeleton,dummy_human_skeleton)

        //Transpose for processing in multiInput
        val transpose_skeleton = humans_skeleton.transpose(3,1,2,0)
        val ret_skeleton = transpose_skeleton.expandDims(axis = 0)
        return ret_skeleton
    }

    private fun harInference(inputData: MultiArray<Float, DN>): String{
        val modelOutput = inferenceOrt(inputData)
        val prob = softmax(modelOutput)
        val label = getLabel(prob)
        return label
    }

    private fun clearSkeletonData(){
        frames_skeleton =
            mk.d3array(144, 25, 3) { 0.0f } // reinitialize landmarks array
    }

    //debug code
    private fun getPoseLandmarksDebugString(poseLandmarks: LandmarkList): String {
        val poseLandmarkStr = """
                Pose landmarks: ${poseLandmarks.landmarkCount}
                
                """.trimIndent()

        Log.v(
            TAG, """
     ======Degree Of Position]======
     nose :${poseLandmarks.landmarkList[0].x},${poseLandmarks.landmarkList[0].y},${poseLandmarks.landmarkList[0].z},${poseLandmarks.landmarkList[0].visibility},

     """.trimIndent()
        )
        Log.v(
            TAG, """
     left ear :${poseLandmarks.landmarkList[7].x},${poseLandmarks.landmarkList[7].y},${poseLandmarks.landmarkList[7].z},${poseLandmarks.landmarkList[7].visibility},

     """.trimIndent()
        )
        Log.v(
            TAG, """
     right ear :${poseLandmarks.landmarkList[8].x},${poseLandmarks.landmarkList[8].y},${poseLandmarks.landmarkList[8].z},${poseLandmarks.landmarkList[8].visibility},

     """.trimIndent()
        )
        Log.v(
            TAG, """
     left hip :${poseLandmarks.landmarkList[23].x},${poseLandmarks.landmarkList[23].y},${poseLandmarks.landmarkList[23].z},${poseLandmarks.landmarkList[23].visibility},

     """.trimIndent()
        )
        Log.v(
            TAG, """
     right ear :${poseLandmarks.landmarkList[24].x},${poseLandmarks.landmarkList[24].y},${poseLandmarks.landmarkList[24].z},${poseLandmarks.landmarkList[24].visibility},

     """.trimIndent()
        )
        return poseLandmarkStr
    }
}