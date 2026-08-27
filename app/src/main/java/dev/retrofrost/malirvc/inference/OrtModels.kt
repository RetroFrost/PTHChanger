package dev.retrofrost.malirvc.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import android.os.Build
import java.io.Closeable
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.util.EnumSet
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

private class AndroidOrtSession(modelPath: String) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val options: OrtSession.SessionOptions
    val session: OrtSession
    val acceleratorEnabled: Boolean

    init {
        var chosenOptions: OrtSession.SessionOptions? = null
        var chosenSession: OrtSession? = null
        var accelerated = false
        if (Build.VERSION.SDK_INT >= 27) {
            try {
                val candidateOptions = OrtSession.SessionOptions()
                candidateOptions.addNnapi(EnumSet.of(NNAPIFlags.USE_FP16))
                try {
                    chosenSession = env.createSession(modelPath, candidateOptions)
                    chosenOptions = candidateOptions
                    accelerated = true
                } catch (e: OrtException) {
                    candidateOptions.close()
                    throw e
                }
            } catch (_: OrtException) {
                // Vendor NNAPI drivers do not necessarily support every op in
                // ContentVec/RMVPE. ORT CPU is the deterministic fallback.
            }
        }
        if (chosenSession == null) {
            chosenOptions = OrtSession.SessionOptions()
            chosenSession = env.createSession(modelPath, chosenOptions)
        }
        options = chosenOptions ?: error("Unable to create ORT options")
        session = chosenSession ?: error("Unable to create ORT session")
        acceleratorEnabled = accelerated
    }

    override fun close() {
        session.close()
        options.close()
    }
}

class ContentVecOrt(modelPath: String) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val runtime = AndroidOrtSession(modelPath)
    private val session get() = runtime.session

    val acceleratorEnabled get() = runtime.acceleratorEnabled

    data class Features(val frames: Int, val channels: Int, val data: FloatArray)

    fun extract(wave16k: FloatArray): Features {
        require(wave16k.isNotEmpty()) { "ContentVec input is empty" }
        require("input_values" in session.inputNames && "attention_mask" in session.inputNames) {
            "Unexpected ContentVec inputs: ${session.inputNames}"
        }

        val mask = LongArray(wave16k.size) { 1L }
        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            inputs["input_values"] = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(wave16k),
                longArrayOf(1, wave16k.size.toLong())
            )
            inputs["attention_mask"] = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(mask),
                longArrayOf(1, wave16k.size.toLong())
            )
            session.run(inputs).use { result ->
                val selected = result.get("hidden_states").orElse(null) as? OnnxTensor
                    ?: (0 until result.size()).asSequence()
                        .mapNotNull { result[it] as? OnnxTensor }
                        .firstOrNull {
                            val shape = it.info.shape
                            shape.size == 3 && shape[0] == 1L && shape[2] == 768L
                        }
                    ?: error("No ContentVec [1,T,768] output found")
                val shape = selected.info.shape
                val buf = selected.floatBuffer ?: error("ContentVec output is not float32")
                val out = FloatArray(buf.remaining())
                buf.get(out)
                return Features(shape[1].toInt(), 768, out)
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    override fun close() = runtime.close()
}

/**
 * RMVPE frontend/decoder matching infer/rmvpe.py from the upstream RVC project:
 * 16 kHz, FFT/window 1024, hop 160, 128 HTK mel bins, 30..8000 Hz,
 * log(clamp(mel, 1e-5)), pad frame count to a multiple of 32, then local
 * weighted-average decoding over the network's 360 pitch bins.
 */
class RmvpeOrt(modelPath: String) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val runtime = AndroidOrtSession(modelPath)
    private val session get() = runtime.session
    private val mel = RmvpeMelSpectrogram()

    val acceleratorEnabled get() = runtime.acceleratorEnabled

    fun extract(wave16k: FloatArray, threshold: Float = 0.03f): FloatArray {
        require("input" in session.inputNames) { "Unexpected RMVPE inputs: ${session.inputNames}" }
        val logMel = mel.compute(wave16k)
        val nFrames = logMel.frames
        val paddedFrames = ((nFrames + 31) / 32) * 32
        val padded = FloatArray(128 * paddedFrames)
        for (m in 0 until 128) {
            System.arraycopy(logMel.data, m * nFrames, padded, m * paddedFrames, nFrames)
        }

        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(padded),
            longArrayOf(1, 128, paddedFrames.toLong())
        ).use { input ->
            session.run(mapOf("input" to input)).use { result ->
                val tensor = result.get("output").orElse(null) as? OnnxTensor
                    ?: result[0] as? OnnxTensor
                    ?: error("RMVPE output is not a tensor")
                val shape = tensor.info.shape
                require(shape.size == 3 && shape[2] == 360L) {
                    "Unexpected RMVPE output shape: ${shape.contentToString()}"
                }
                val buf = tensor.floatBuffer ?: error("RMVPE output is not float32")
                val salience = FloatArray(buf.remaining())
                buf.get(salience)
                val networkFrames = shape[1].toInt()
                require(networkFrames >= nFrames) {
                    "RMVPE returned only $networkFrames frames for $nFrames-frame input"
                }
                return decode(salience, networkFrames, nFrames, threshold)
            }
        }
    }

    private fun decode(
        salience: FloatArray,
        networkFrames: Int,
        requestedFrames: Int,
        threshold: Float
    ): FloatArray {
        val result = FloatArray(requestedFrames)
        for (frame in 0 until requestedFrames) {
            val offset = frame * 360
            var center = 0
            var peak = salience[offset]
            for (bin in 1 until 360) {
                val value = salience[offset + bin]
                if (value > peak) {
                    peak = value
                    center = bin
                }
            }
            if (peak <= threshold) {
                result[frame] = 0f
                continue
            }
            var weighted = 0.0
            var weight = 0.0
            val from = max(0, center - 4)
            val to = minOf(359, center + 4)
            for (bin in from..to) {
                val s = salience[offset + bin].toDouble()
                val cents = 20.0 * bin + 1997.3794084376191
                weighted += s * cents
                weight += s
            }
            if (weight <= 0.0) {
                result[frame] = 0f
            } else {
                val cents = weighted / weight
                result[frame] = (10.0 * 2.0.pow(cents / 1200.0)).toFloat()
            }
        }
        return result
    }

    override fun close() = runtime.close()
}

private data class MelResult(val frames: Int, val data: FloatArray)

private class RmvpeMelSpectrogram {
    private companion object {
        const val SAMPLE_RATE = 16_000
        const val FFT_SIZE = 1024
        const val HOP = 160
        const val MELS = 128
        const val FMIN = 30.0
        const val FMAX = 8000.0
        const val CLAMP = 1e-5
    }

    // torch.hann_window(1024) uses a periodic Hann window.
    private val window = FloatArray(FFT_SIZE) { n ->
        (0.5 - 0.5 * cos(2.0 * PI * n / FFT_SIZE)).toFloat()
    }
    private val melBasis = makeMelBasis()

    fun compute(audio: FloatArray): MelResult {
        require(audio.size > FFT_SIZE / 2) { "Audio is too short for RMVPE" }
        // torch.stft(... center=true) reflect-pads n_fft/2 on each side.
        val frames = audio.size / HOP + 1
        val magnitude = FloatArray(FFT_SIZE / 2 + 1)
        val real = DoubleArray(FFT_SIZE)
        val imag = DoubleArray(FFT_SIZE)
        val out = FloatArray(MELS * frames)
        val half = FFT_SIZE / 2

        for (frame in 0 until frames) {
            val paddedStart = frame * HOP
            for (n in 0 until FFT_SIZE) {
                val originalIndex = paddedStart + n - half
                val sample = audio[reflectIndex(originalIndex, audio.size)].toDouble()
                real[n] = sample * window[n]
                imag[n] = 0.0
            }
            fftInPlace(real, imag)
            for (k in 0..half) {
                magnitude[k] = sqrt(real[k] * real[k] + imag[k] * imag[k]).toFloat()
            }
            for (m in 0 until MELS) {
                var sum = 0.0
                val basisOffset = m * (half + 1)
                for (k in 0..half) {
                    sum += melBasis[basisOffset + k] * magnitude[k]
                }
                out[m * frames + frame] = ln(max(CLAMP, sum)).toFloat()
            }
        }
        return MelResult(frames, out)
    }

    private fun reflectIndex(index: Int, length: Int): Int {
        var i = index
        while (i < 0 || i >= length) {
            i = if (i < 0) -i else 2 * length - 2 - i
        }
        return i
    }

    /** librosa.filters.mel(... htk=true), default Slaney area normalization. */
    private fun makeMelBasis(): FloatArray {
        val fftBins = FFT_SIZE / 2 + 1
        val melMin = hzToHtkMel(FMIN)
        val melMax = hzToHtkMel(FMAX)
        val melPoints = DoubleArray(MELS + 2) { i ->
            melMin + (melMax - melMin) * i / (MELS + 1).toDouble()
        }
        val hzPoints = DoubleArray(MELS + 2) { i -> htkMelToHz(melPoints[i]) }
        val fftFreqs = DoubleArray(fftBins) { k -> k.toDouble() * SAMPLE_RATE / FFT_SIZE }
        val basis = FloatArray(MELS * fftBins)
        for (m in 0 until MELS) {
            val lower = hzPoints[m]
            val center = hzPoints[m + 1]
            val upper = hzPoints[m + 2]
            val enorm = 2.0 / (upper - lower)
            for (k in 0 until fftBins) {
                val f = fftFreqs[k]
                val lowerSlope = (f - lower) / (center - lower)
                val upperSlope = (upper - f) / (upper - center)
                val triangle = max(0.0, minOf(lowerSlope, upperSlope))
                basis[m * fftBins + k] = (triangle * enorm).toFloat()
            }
        }
        return basis
    }

    private fun hzToHtkMel(hz: Double): Double = 2595.0 * log10(1.0 + hz / 700.0)
    private fun htkMelToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private fun fftInPlace(real: DoubleArray, imag: DoubleArray) {
        val n = real.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j xor bit
            if (i < j) {
                val tr = real[i]
                real[i] = real[j]
                real[j] = tr
                val ti = imag[i]
                imag[i] = imag[j]
                imag[j] = ti
            }
        }

        var len = 2
        while (len <= n) {
            val angle = -2.0 * PI / len
            val wLenReal = cos(angle)
            val wLenImag = sin(angle)
            var start = 0
            while (start < n) {
                var wReal = 1.0
                var wImag = 0.0
                val half = len / 2
                for (k in 0 until half) {
                    val even = start + k
                    val odd = even + half
                    val oddReal = real[odd] * wReal - imag[odd] * wImag
                    val oddImag = real[odd] * wImag + imag[odd] * wReal
                    val evenReal = real[even]
                    val evenImag = imag[even]
                    real[even] = evenReal + oddReal
                    imag[even] = evenImag + oddImag
                    real[odd] = evenReal - oddReal
                    imag[odd] = evenImag - oddImag
                    val nextWReal = wReal * wLenReal - wImag * wLenImag
                    wImag = wReal * wLenImag + wImag * wLenReal
                    wReal = nextWReal
                }
                start += len
            }
            len = len shl 1
        }
    }
}
