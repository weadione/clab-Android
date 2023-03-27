package com.example.newpaintlapse

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Color
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.TextView
import com.example.mlkitpaintlapse.R
import com.google.mlkit.vision.pose.Pose
import com.google.mlkit.vision.pose.PoseLandmark
import org.jetbrains.kotlinx.multik.api.*
import org.jetbrains.kotlinx.multik.ndarray.data.*
import org.jetbrains.kotlinx.multik.ndarray.operations.*
import java.nio.FloatBuffer
import java.util.*
import kotlin.collections.ArrayList
import kotlin.concurrent.fixedRateTimer
import kotlin.math.exp
import kotlin.math.pow
import kotlin.math.sqrt

data class WorldPose(val x: Float, val y: Float, val z: Float, val visibility: Float)
enum class RatioAnchor{SHOULDER, EAR, HIP, EYE}

class HARHelper(val context: Context) {
    //skeletonData val
    private var frames_skeleton = mk.d3array(50,25,3){0.0f}
    private var skeletonBuffer =  ArrayList<D2Array<Float>>()
    private lateinit var timer: Timer


    //onnxruntime val
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    //label val
    private var label: String = ""
    private var previousLabel = 0
    private var updatedLabel = 0
    private var labelColor = Color.WHITE
    private lateinit var harLabel: TextView

    //debug val
    private var prevSamplingTime: Long = 0
    private var halfWidth: Int = 0
    private var halfHeight: Int = 0
    private val anchorData = arrayOf(arrayOf(11,12),arrayOf(7,8),arrayOf(23,24),arrayOf(3,6))
    private val ratioData = arrayOf(0.325f, 0.148f, 0.221f, 0.0325f)
    private var anchor = RatioAnchor.SHOULDER

    public fun initOrt(){
        ortEnv = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL)
        ortSession = CreateOrtSession()
    }


    private fun CreateOrtSession(): OrtSession? {
        val so = OrtSession.SessionOptions()
        return ortEnv?.createSession(readModel(), so)
    }

    private fun readModel(): ByteArray {
        return  context.resources.openRawResource(R.raw.har_gcn).readBytes()
    }

    fun setSize(screenWidth: Int, screenHeight: Int){
        halfWidth = screenWidth/2
        halfHeight = screenHeight/2
    }

    private fun getSkelton() {
        val resString = harInference(convertSkeletonData())
        Log.v("label:", label)
        Log.v("time:", (SystemClock.uptimeMillis() - prevSamplingTime).toString())

        Handler(Looper.getMainLooper()).post(){
            harLabel.setTextColor(labelColor)
            harLabel.setText(resString)
        }
        clearSkeletonData()
        prevSamplingTime = SystemClock.uptimeMillis()

    }

    public fun saveSkeletonData(pose: Pose, orentation: Int){
        var one_frame_skeleton = mk.d2array(25,3){0.0f}
        val landmark = local2WolrdLandmark(pose.allPoseLandmarks)
        val rotation = calculateOrientation(orentation)

        for(i: Int in 0..24) {
            if (landmark[i].visibility >= 0.75) {
                when(rotation){
                    0 ->{
                        one_frame_skeleton[i, 0] = landmark[i].x
                        one_frame_skeleton[i, 1] = landmark[i].y
                    }
                    1 ->{
                        one_frame_skeleton[i, 0] = -landmark[i].y
                        one_frame_skeleton[i, 1] = landmark[i].x
                    }
                    2 -> {
                        one_frame_skeleton[i, 0] = -landmark[i].x
                        one_frame_skeleton[i, 1] = -landmark[i].y
                    }
                    3-> {
                        one_frame_skeleton[i, 0] = landmark[i].y
                        one_frame_skeleton[i, 1] = -landmark[i].x
                    }
                }
                one_frame_skeleton[i, 2] = landmark[i].z
            }
        }
        skeletonBuffer.add(one_frame_skeleton)
//        for(i in 0 until 25){
//            Log.v("landmark","${i}: ${one_frame_skeleton[i][0]}, ${one_frame_skeleton[i][1]}, ${one_frame_skeleton[i][2]}")
//        }
        Log.v("landmark","${one_frame_skeleton[0][0]}, ${one_frame_skeleton[0][1]}, ${one_frame_skeleton[0][2]}")
//         Log.v("","========================================================================================")
    }

    private fun local2WolrdLandmark(poseLandmark: List<PoseLandmark>): List<WorldPose>{
        val numLandmarks = 25
        val worldPoses = mutableListOf<WorldPose>()

        val leftHip = poseLandmark[PoseLandmark.LEFT_HIP]
        val rightHip = poseLandmark[PoseLandmark.RIGHT_HIP]
        val centerHip = WorldPose(
            (leftHip.position3D.x+rightHip.position3D.x)/2,
            (leftHip.position3D.y+rightHip.position3D.y)/2,
            (leftHip.position3D.z+rightHip.position3D.z)/2,
            (leftHip.inFrameLikelihood+rightHip.inFrameLikelihood)/2
        )

        for(i in RatioAnchor.values()){
            if((poseLandmark[anchorData[i.ordinal][0]].inFrameLikelihood+poseLandmark[anchorData[i.ordinal][1]].inFrameLikelihood)/2.0 > 0.75){
                anchor = i
//                Log.e("anchor",i.ordinal.toString())
                break
            }
            if(i == RatioAnchor.EYE)
                Log.e("ratio","앵커가 없습니다.")
        }

        val distance = sqrt(
            (poseLandmark[anchorData[anchor.ordinal][0]].position3D.x - poseLandmark[anchorData[anchor.ordinal][1]].position3D.x).pow(2)
                    + (poseLandmark[anchorData[anchor.ordinal][0]].position3D.y - poseLandmark[anchorData[anchor.ordinal][1]].position3D.y).pow(2)
                    + (poseLandmark[anchorData[anchor.ordinal][0]].position3D.z - poseLandmark[anchorData[anchor.ordinal][1]].position3D.z).pow(2)
        )
        val distRatio = ratioData[anchor.ordinal] / distance


        Log.v("ratio","${distRatio}, ${anchor}")

        for(i in 0 until numLandmarks){
            val x = (poseLandmark[i].position3D.x-centerHip.x)*distRatio
            val y = (poseLandmark[i].position3D.y-centerHip.y)*distRatio
            val z = (poseLandmark[i].position3D.z-centerHip.z)*distRatio
            worldPoses.add(WorldPose(x,y,z,poseLandmark[i].inFrameLikelihood))
        }

        return worldPoses
    }

    private fun calculateOrientation(orentation: Int) = when (orentation) {
       in 45 until 135 -> 1  // 90
       in 135 until 225 -> 2 // 180
       in 225 until 315 -> 3 //270
       else -> 0  //portrait(0)
    }


    private fun _sampleSkeletonData(){
        val curskeletonBuffer = ArrayList<D2Array<Float>>()
        curskeletonBuffer.addAll(skeletonBuffer)
        skeletonBuffer.clear()

        val frameNum = curskeletonBuffer.size
        Log.v("num",frameNum.toString())
        if(frameNum>=40) {
            val skipInterval = frameNum / 40.0
            for (i in 0 until 40)
                frames_skeleton[i] = curskeletonBuffer[(i * skipInterval).toInt()]
        }
        else
            for(i in 0 until frameNum)
                frames_skeleton[i] = curskeletonBuffer[i]
    }


    private fun convertSkeletonData() : MultiArray<Float, DN>{
        //dummy humman skeleton data to align the input dimension of the model
        _sampleSkeletonData()
        val ret_skeleton = frames_skeleton.expandDims(axis = 0).transpose(3,1,2,0).expandDims(axis = 0)
        return ret_skeleton
    }

    private fun clearSkeletonData(){
        frames_skeleton =
            mk.d3array(50, 25, 3) { 0.0f } // reinitialize landmarks array
    }

    private fun harInference(inputData: MultiArray<Float, DN>): String{
        val modelOutput = inferenceOrt(inputData)
        val prob = softmax(modelOutput)
        label = getLabel(prob)
        return label
    }

    fun inferenceOrt(inputData: MultiArray<Float, DN>): FloatArray{
        val inputName = ortSession?.inputNames?.iterator()?.next()
        val shape = longArrayOf(1, 3, 50, 25, 1)
        val ortEnv = OrtEnvironment.getEnvironment()
        val floatArrayData = inputData.toFloatArray()
        val buffer = FloatBuffer.allocate(floatArrayData.size)
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

            if(updatedLabel!=top1Index && previousLabel == top1Index){
                updatedLabel = top1Index
            }
            previousLabel = top1Index

            top1Index = updatedLabel

            if(top1Index==1)
                labelColor = Color.RED
            else
                labelColor = Color.GREEN

            when (top1Index) {
                0 -> ""
                1 -> "Painting:" + String.format("%.1f", (big3[top1Index] * 100)) + "%"
                else -> "Interview:" + String.format("%.1f", (big3[top1Index] * 100)) + "%"
            }
        } catch (e: Exception) {
            e.message?.let { Log.v("HAR", it) }
            ""
        }
    }

    public fun startSkeletonTimer(harTextView: TextView){
        harLabel = harTextView
        val skeletonTimer = fixedRateTimer(name="SkeletonTimer", initialDelay = 0L, period = 4000L){
            getSkelton()
        }
        timer = skeletonTimer
    }

    public fun stopSkeletonTimer(){
        timer.cancel()
    }

}