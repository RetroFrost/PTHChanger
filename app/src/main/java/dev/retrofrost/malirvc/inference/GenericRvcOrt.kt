package dev.retrofrost.malirvc.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.os.Build
import dev.retrofrost.malirvc.model.PthCheckpoint
import org.json.JSONObject
import java.io.Closeable
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet
import java.util.Random

/**
 * Voice-neutral RVC v2 synthesizer.
 *
 * The APK contains the architecture as ONNX. Every trainable initializer is
 * replaced before session creation with the matching tensor from the selected
 * RVC .pth checkpoint. The user's voice never needs to be converted to a
 * second model format.
 */
class GenericRvcOrt(
    modelPath: String,
    manifestJson: String,
    private val checkpoint: PthCheckpoint
) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val initializerTensors = ArrayList<OnnxTensor>()
    private val options: OrtSession.SessionOptions
    private val session: OrtSession

    /** True when the session was created with Android NNAPI enabled. */
    val acceleratorEnabled: Boolean

    init {
        val manifest = JSONObject(manifestJson)
        require(manifest.getString("version") == "v2")
        require(manifest.getBoolean("f0"))
        require(manifest.getInt("sampleRate") == checkpoint.info.sampleRate)
        require(manifest.getInt("speakerCount") == checkpoint.info.speakerCount) {
            "Model has ${checkpoint.info.speakerCount} speaker embeddings, but this runtime profile expects ${manifest.getInt("speakerCount")}"
        }

        val specs = manifest.getJSONArray("weights")
        for (i in 0 until specs.length()) {
            val spec = specs.getJSONObject(i)
            val checkpointName = spec.getString("checkpointName")
            val onnxName = spec.getString("onnxName")
            val expected = spec.getJSONArray("shape")
            val actual = checkpoint.weights[checkpointName]
                ?: error("Checkpoint is missing required RVC tensor '$checkpointName'")
            require(actual.shape.size == expected.length()) {
                "Rank mismatch for $checkpointName"
            }
            for (d in actual.shape.indices) {
                require(actual.shape[d] == expected.getLong(d)) {
                    "Shape mismatch for $checkpointName: ${actual.shape.contentToString()}"
                }
            }
            val values = checkpoint.readTensorAsFloat(checkpointName)
            initializerTensors += OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(values),
                actual.shape
            )
        }

        fun makeOptions(useNnapi: Boolean): OrtSession.SessionOptions {
            val result = OrtSession.SessionOptions()
            for (i in 0 until specs.length()) {
                result.addInitializer(specs.getJSONObject(i).getString("onnxName"), initializerTensors[i])
            }
            if (useNnapi && Build.VERSION.SDK_INT >= 27) {
                result.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
            }
            return result
        }

        var selectedOptions: OrtSession.SessionOptions? = null
        var selectedSession: OrtSession? = null
        var accelerated = false
        if (Build.VERSION.SDK_INT >= 27) {
            try {
                selectedOptions = makeOptions(true)
                selectedSession = env.createSession(modelPath, selectedOptions)
                accelerated = true
            } catch (_: OrtException) {
                selectedOptions?.close()
                selectedOptions = null
            }
        }
        if (selectedSession == null) {
            selectedOptions = makeOptions(false)
            selectedSession = env.createSession(modelPath, selectedOptions)
        }
        options = selectedOptions
        session = selectedSession
        acceleratorEnabled = accelerated
    }

    fun infer(
        phone: FloatArray,
        frames: Int,
        pitch: LongArray,
        pitchf: FloatArray,
        speakerId: Long
    ): FloatArray {
        require(phone.size == frames * 768)
        require(pitch.size == frames)
        require(pitchf.size == frames)

        val rnd = FloatArray(192 * frames)
        val random = Random()
        for (i in rnd.indices) rnd[i] = random.nextGaussian().toFloat()

        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            inputs["phone"] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(phone), longArrayOf(1, frames.toLong(), 768)
            )
            inputs["phone_lengths"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(frames.toLong())), longArrayOf(1)
            )
            inputs["pitch"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(pitch), longArrayOf(1, frames.toLong())
            )
            inputs["pitchf"] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(pitchf), longArrayOf(1, frames.toLong())
            )
            inputs["sid"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(speakerId)), longArrayOf(1)
            )
            inputs["rnd"] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(rnd), longArrayOf(1, 192, frames.toLong())
            )

            session.run(inputs).use { result ->
                val tensor = result[0] as? OnnxTensor ?: error("RVC synthesizer output is not a tensor")
                val buffer = tensor.floatBuffer ?: error("RVC synthesizer output is not float32")
                val out = FloatArray(buffer.remaining())
                buffer.get(out)
                return out
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    override fun close() {
        session.close()
        options.close()
        initializerTensors.forEach { it.close() }
    }
}
