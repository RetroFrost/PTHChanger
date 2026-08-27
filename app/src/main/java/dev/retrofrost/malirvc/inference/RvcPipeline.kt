package dev.retrofrost.malirvc.inference

import android.content.Context
import android.net.Uri
import dev.retrofrost.malirvc.audio.AndroidAudioDecoder
import dev.retrofrost.malirvc.audio.LinearResampler
import dev.retrofrost.malirvc.audio.WavWriter
import dev.retrofrost.malirvc.model.PthCheckpoint
import dev.retrofrost.malirvc.util.AssetUtils
import java.io.File
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.roundToLong

class RvcPipeline(private val context: Context) {
    companion object {
        const val CONTENTVEC = "runtime/contentvec_v2.onnx"
        const val RMVPE = "runtime/rmvpe.onnx"
        const val SYNTH = "runtime/rvc_v2_f0_40k_s109.onnx"
        const val MANIFEST = "runtime/rvc_weights_manifest.json"

        fun missingRuntimeAssets(context: Context): List<String> =
            listOf(CONTENTVEC, RMVPE, SYNTH, MANIFEST).filterNot { AssetUtils.exists(context, it) }
    }

    suspend fun convert(
        checkpointFile: File,
        inputAudio: Uri,
        pitchShift: Int,
        speakerId: Int,
        onStatus: (String) -> Unit
    ): File {
        onStatus("Reading .pth model")
        PthCheckpoint(checkpointFile).use { checkpoint ->
            require(checkpoint.info.version == "v2" && checkpoint.info.f0) {
                "Only F0-enabled RVC v2 checkpoints are supported"
            }
            require(checkpoint.info.sampleRate == 40_000) {
                "This runtime profile is 40 kHz; model is ${checkpoint.info.sampleRate} Hz"
            }
            require(checkpoint.info.speakerCount == 109) {
                "This runtime profile expects 109 speaker embeddings; model has ${checkpoint.info.speakerCount}"
            }
            require(speakerId in 0 until checkpoint.info.speakerCount) { "Speaker ID out of range" }

            val missing = missingRuntimeAssets(context)
            require(missing.isEmpty()) {
                "This APK is missing bundled runtime assets: ${missing.joinToString()}"
            }

            onStatus("Decoding audio")
            val decoded = AndroidAudioDecoder.decodeMono(context, inputAudio)
            val wave16k = LinearResampler.resample(decoded.samples, decoded.sampleRate, 16_000)
            require(wave16k.size >= 1600) { "Audio is too short" }

            val contentVecFile = AssetUtils.copyToFiles(context, CONTENTVEC)
            val rmvpeFile = AssetUtils.copyToFiles(context, RMVPE)
            val synthFile = AssetUtils.copyToFiles(context, SYNTH)
            val manifest = context.assets.open(MANIFEST).bufferedReader().use { it.readText() }

            val chunkSamples = 20 * 16_000
            val nominalOverlap = 4_000
            val starts = chunkStarts(wave16k.size, chunkSamples, nominalOverlap)

            val audio40k = ContentVecOrt(contentVecFile.absolutePath).use { contentVec ->
                RmvpeOrt(rmvpeFile.absolutePath).use { rmvpe ->
                    GenericRvcOrt(synthFile.absolutePath, manifest, checkpoint).use { synth ->
                        var merged = FloatArray(0)
                        var previousEnd16k = 0
                        starts.forEachIndexed { index, start16k ->
                            val end16k = (start16k + chunkSamples).coerceAtMost(wave16k.size)
                            val chunk = wave16k.copyOfRange(start16k, end16k)
                            onStatus("Converting ${index + 1}/${starts.size} • ContentVec")
                            val baseFeatures = contentVec.extract(chunk)
                            val pLen = chunk.size / 160
                            require(pLen > 0) { "Audio chunk is too short" }
                            val phone = upsampleContentVec(baseFeatures, pLen)

                            onStatus("Converting ${index + 1}/${starts.size} • RMVPE")
                            var f0 = rmvpe.extract(chunk)
                            f0 = alignF0(f0, pLen)
                            val ratio = 2.0.pow(pitchShift / 12.0).toFloat()
                            for (i in f0.indices) if (f0[i] > 0f) f0[i] *= ratio
                            val coarse = coarsePitch(f0)

                            val backend = if (synth.acceleratorEnabled) "Android accelerator" else "CPU"
                            onStatus("Converting ${index + 1}/${starts.size} • $backend")
                            val converted = synth.infer(phone, pLen, coarse, f0, speakerId.toLong())
                            val overlap16k = if (index == 0) 0 else (previousEnd16k - start16k).coerceAtLeast(0)
                            val overlap40k = ((overlap16k.toLong() * 40_000L) / 16_000L).toInt()
                            merged = crossfadeAppend(merged, converted, overlap40k)
                            previousEnd16k = end16k
                        }
                        merged
                    }
                }
            }

            onStatus("Writing WAV")
            val outDir = File(context.cacheDir, "converted").apply { mkdirs() }
            val safeName = checkpoint.info.modelName.replace(Regex("[^A-Za-z0-9._-]+"), "_")
            val out = File(outDir, "${safeName}_${System.currentTimeMillis()}.wav")
            WavWriter.write16BitMono(out, audio40k, 40_000)
            onStatus("Done")
            return out
        }
    }

    private fun upsampleContentVec(features: ContentVecOrt.Features, targetFrames: Int): FloatArray {
        require(features.channels == 768 && features.frames > 0)
        val out = FloatArray(targetFrames * 768)
        for (t in 0 until targetFrames) {
            val src = (t / 2).coerceAtMost(features.frames - 1)
            System.arraycopy(features.data, src * 768, out, t * 768, 768)
        }
        return out
    }

    private fun alignF0(source: FloatArray, target: Int): FloatArray {
        if (source.size == target) return source
        if (source.isEmpty()) return FloatArray(target)
        val out = FloatArray(target)
        val scale = if (target <= 1) 0.0 else (source.size - 1).toDouble() / (target - 1).toDouble()
        for (i in out.indices) {
            val pos = i * scale
            val a = pos.toInt().coerceIn(0, source.lastIndex)
            val b = (a + 1).coerceAtMost(source.lastIndex)
            val f = (pos - a).toFloat()
            out[i] = source[a] * (1f - f) + source[b] * f
        }
        return out
    }

    private fun chunkStarts(total: Int, chunk: Int, overlap: Int): List<Int> {
        if (total <= chunk) return listOf(0)
        val step = chunk - overlap
        val starts = ArrayList<Int>()
        var start = 0
        starts += 0
        while (start + chunk < total) {
            var next = start + step
            if (next + chunk >= total) next = total - chunk
            if (next <= start) break
            starts += next
            start = next
        }
        return starts
    }

    private fun crossfadeAppend(a: FloatArray, b: FloatArray, requestedOverlap: Int): FloatArray {
        if (a.isEmpty()) return b
        val overlap = requestedOverlap.coerceAtMost(a.size).coerceAtMost(b.size)
        if (overlap <= 0) return a + b
        val out = FloatArray(a.size + b.size - overlap)
        System.arraycopy(a, 0, out, 0, a.size - overlap)
        val base = a.size - overlap
        for (i in 0 until overlap) {
            val t = if (overlap == 1) 1f else i.toFloat() / (overlap - 1).toFloat()
            out[base + i] = a[base + i] * (1f - t) + b[i] * t
        }
        System.arraycopy(b, overlap, out, a.size, b.size - overlap)
        return out
    }

    private fun coarsePitch(f0: FloatArray): LongArray {
        val minMel = 1127.0 * ln(1.0 + 50.0 / 700.0)
        val maxMel = 1127.0 * ln(1.0 + 1100.0 / 700.0)
        return LongArray(f0.size) { i ->
            val hz = f0[i].toDouble()
            if (hz <= 0.0) 1L else {
                val mel = 1127.0 * ln(1.0 + hz / 700.0)
                (((mel - minMel) * 254.0 / (maxMel - minMel)) + 1.0)
                    .coerceIn(1.0, 255.0)
                    .roundToLong()
            }
        }
    }
}
