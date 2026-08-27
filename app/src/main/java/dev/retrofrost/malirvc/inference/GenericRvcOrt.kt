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
 * Voice-neutral RVC v2 synthesizer whose 457 initializers are replaced by the
 * tensors parsed directly from the selected .pth checkpoint.
 *
 * The graph is intentionally fixed-width. Upstream RVC's relative-attention
 * implementation bakes sequence lengths when legacy ONNX tracing is used, so
 * pretending that graph is dynamic is incorrect. We instead pad every chunk
 * to [fixedFrames] and crop the resulting audio to the real frame count.
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
    private val fixedFrames: Int
    private val samplesPerFrame: Int

    /** True means NNAPI was added successfully; individual node placement is driver-controlled. */
    val acceleratorEnabled: Boolean

    val maxFrames: Int get() = fixedFrames

    init {
        val manifest = JSONObject(manifestJson)
        require(manifest.getInt("format") >= 2) { "Unsupported RVC runtime manifest" }
        require(manifest.getString("version") == "v2")
        require(manifest.getBoolean("f0"))
        require(manifest.getInt("sampleRate") == checkpoint.info.sampleRate)
        require(manifest.getInt("speakerCount") == checkpoint.info.speakerCount) {
            "Model has ${checkpoint.info.speakerCount} speaker embeddings, but this runtime profile expects ${manifest.getInt("speakerCount")}"
        }
        fixedFrames = manifest.getInt("fixedFrames")
        samplesPerFrame = manifest.getInt("samplesPerFrame")
        require(fixedFrames > 0 && samplesPerFrame > 0)

        val specs = manifest.getJSONArray("weights")
        require(specs.length() == 457) { "RVC runtime manifest has ${specs.length()} weights, expected 457" }
        for (i in 0 until specs.length()) {
            val spec = specs.getJSONObject(i)
            val checkpointName = spec.getString("checkpointName")
            val expected = spec.getJSONArray("shape")
            val actual = checkpoint.weights[checkpointName]
                ?: error("Checkpoint is missing required RVC tensor '$checkpointName'")
            require(actual.shape.size == expected.length()) { "Rank mismatch for $checkpointName" }
            for (d in actual.shape.indices) {
                require(actual.shape[d] == expected.getLong(d)) {
                    "Shape mismatch for $checkpointName: ${actual.shape.contentToString()}"
                }
            }
            val values = checkpoint.readTensorAsFloat(checkpointName)
            initializerTensors += OnnxTensor.createTensor(env, FloatBuffer.wrap(values), actual.shape)
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
                val acceleratedOptions = makeOptions(true)
                try {
                    selectedSession = env.createSession(modelPath, acceleratedOptions)
                    selectedOptions = acceleratedOptions
                    accelerated = true
                } catch (e: OrtException) {
                    acceleratedOptions.close()
                    throw e
                }
            } catch (_: OrtException) {
                // A device may reject this graph for NNAPI. ORT CPU stays a fully offline fallback.
            }
        }
        if (selectedSession == null) {
            selectedOptions = makeOptions(false)
            selectedSession = env.createSession(modelPath, selectedOptions)
        }
        options = selectedOptions ?: error("Failed to create ONNX Runtime options")
        session = selectedSession ?: error("Failed to create RVC ONNX Runtime session")
        acceleratorEnabled = accelerated
    }

    fun infer(
        phone: FloatArray,
        frames: Int,
        pitch: LongArray,
        pitchf: FloatArray,
        speakerId: Long
    ): FloatArray {
        require(frames in 1..fixedFrames) { "RVC chunk has $frames frames; max is $fixedFrames" }
        require(phone.size == frames * 768)
        require(pitch.size == frames)
        require(pitchf.size == frames)

        val paddedPhone = FloatArray(fixedFrames * 768)
        System.arraycopy(phone, 0, paddedPhone, 0, phone.size)
        val paddedPitch = LongArray(fixedFrames) { 1L }
        System.arraycopy(pitch, 0, paddedPitch, 0, pitch.size)
        val paddedPitchf = FloatArray(fixedFrames)
        System.arraycopy(pitchf, 0, paddedPitchf, 0, pitchf.size)

        val rnd = FloatArray(192 * fixedFrames)
        val random = Random()
        for (channel in 0 until 192) {
            val base = channel * fixedFrames
            for (i in 0 until frames) rnd[base + i] = random.nextGaussian().toFloat()
        }

        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            inputs["phone"] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(paddedPhone), longArrayOf(1, fixedFrames.toLong(), 768)
            )
            inputs["phone_lengths"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(frames.toLong())), longArrayOf(1)
            )
            inputs["pitch"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(paddedPitch), longArrayOf(1, fixedFrames.toLong())
            )
            inputs["pitchf"] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(paddedPitchf), longArrayOf(1, fixedFrames.toLong())
            )
            inputs["sid"] = OnnxTensor.createTensor(
                env, LongBuffer.wrap(longArrayOf(speakerId)), longArrayOf(1)
            )
            inputs["rnd"] = OnnxTensor.createTensor(
                env, FloatBuffer.wrap(rnd), longArrayOf(1, 192, fixedFrames.toLong())
            )

            session.run(inputs).use { result ->
                val tensor = result[0] as? OnnxTensor ?: error("RVC synthesizer output is not a tensor")
                val buffer = tensor.floatBuffer ?: error("RVC synthesizer output is not float32")
                val full = FloatArray(buffer.remaining())
                buffer.get(full)
                val wanted = frames * samplesPerFrame
                require(full.size >= wanted) { "RVC output is shorter than expected (${full.size} < $wanted)" }
                return full.copyOf(wanted)
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
