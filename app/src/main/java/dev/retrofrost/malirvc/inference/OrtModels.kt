package dev.retrofrost.malirvc.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.Closeable
import java.nio.FloatBuffer

class ContentVecOrt(modelPath: String) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelPath, OrtSession.SessionOptions())

    data class Features(val frames: Int, val channels: Int, val data: FloatArray)

    fun extract(wave16k: FloatArray): Features {
        require(session.inputNames.size == 1) {
            "contentvec_v2.onnx must expose one waveform input; got ${session.inputNames}"
        }
        val inputName = session.inputNames.first()
        OnnxTensor.createTensor(env, FloatBuffer.wrap(wave16k), longArrayOf(1, wave16k.size.toLong())).use { input ->
            session.run(mapOf(inputName to input)).use { result ->
                var tensor: OnnxTensor? = null
                for (i in 0 until result.size()) {
                    val candidate = result[i] as? OnnxTensor ?: continue
                    val shape = candidate.info.shape
                    if (shape.size == 3 && shape[0] == 1L && shape[2] == 768L) {
                        tensor = candidate
                        break
                    }
                }
                val selected = tensor ?: error("No ContentVec [1,T,768] output found")
                val shape = selected.info.shape
                val buf = selected.floatBuffer ?: error("ContentVec output is not float")
                val out = FloatArray(buf.remaining())
                buf.get(out)
                return Features(shape[1].toInt(), 768, out)
            }
        }
    }

    override fun close() = session.close()
}

class RmvpeOrt(modelPath: String) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session = env.createSession(modelPath, OrtSession.SessionOptions())

    fun extract(wave16k: FloatArray, threshold: Float = 0.03f): FloatArray {
        val names = session.inputNames.toList()
        require(names.isNotEmpty()) { "RMVPE model has no inputs" }
        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            inputs[names[0]] = OnnxTensor.createTensor(env, FloatBuffer.wrap(wave16k), longArrayOf(1, wave16k.size.toLong()))
            if (names.size >= 2) {
                inputs[names[1]] = OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(threshold)), longArrayOf(1))
            }
            session.run(inputs).use { result ->
                val tensor = result[0] as? OnnxTensor ?: error("RMVPE output is not a tensor")
                val buf = tensor.floatBuffer ?: error("RMVPE output is not float")
                val out = FloatArray(buf.remaining())
                buf.get(out)
                return out
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    override fun close() = session.close()
}
