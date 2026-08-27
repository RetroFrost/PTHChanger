package dev.retrofrost.malirvc.inference

import dev.retrofrost.malirvc.model.PthCheckpoint
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.Closeable
import java.io.File
import java.util.Random

/**
 * Executes a generic RVC-v2 graph. All synthesizer parameters are user inputs,
 * so the selected .pth supplies the weights at runtime. The .pte itself has no
 * voice baked into it.
 */
class GenericRvcExecuTorch(
    pteFile: File,
    manifestJson: String,
    checkpoint: PthCheckpoint
) : Closeable {
    private val module = Module.load(pteFile.absolutePath)
    private val weightInputs: Array<EValue>
    private val userInputStart: Int

    init {
        val manifest = JSONObject(manifestJson)
        userInputStart = manifest.getInt("userInputStart")
        val specs = manifest.getJSONArray("weights")
        require(specs.length() == userInputStart)
        weightInputs = Array(specs.length()) { i ->
            val spec = specs.getJSONObject(i)
            val name = spec.getString("name")
            val expected = spec.getJSONArray("shape")
            val actual = checkpoint.weights[name] ?: error("Checkpoint is missing '$name'")
            require(actual.shape.size == expected.length()) { "Shape mismatch for $name" }
            for (d in actual.shape.indices) {
                require(actual.shape[d] == expected.getLong(d)) {
                    "Shape mismatch for $name: ${actual.shape.contentToString()}"
                }
            }
            val data = checkpoint.readTensorAsFloat(name)
            EValue.from(Tensor.fromBlob(data, actual.shape))
        }
    }

    fun infer(
        phone: FloatArray,
        frames: Int,
        pitch: LongArray,
        pitchf: FloatArray,
        speakerId: Long
    ): FloatArray {
        require(phone.size == frames * 768)
        require(pitch.size == frames && pitchf.size == frames)

        val rnd = FloatArray(192 * frames)
        val random = Random()
        for (i in rnd.indices) rnd[i] = random.nextGaussian().toFloat()

        val dynamic = arrayOf(
            EValue.from(Tensor.fromBlob(phone, longArrayOf(1, frames.toLong(), 768))),
            EValue.from(Tensor.fromBlob(longArrayOf(frames.toLong()), longArrayOf(1))),
            EValue.from(Tensor.fromBlob(pitch, longArrayOf(1, frames.toLong()))),
            EValue.from(Tensor.fromBlob(pitchf, longArrayOf(1, frames.toLong()))),
            EValue.from(Tensor.fromBlob(longArrayOf(speakerId), longArrayOf(1))),
            EValue.from(Tensor.fromBlob(rnd, longArrayOf(1, 192, frames.toLong())))
        )
        val all = arrayOfNulls<EValue>(weightInputs.size + dynamic.size)
        for (i in weightInputs.indices) all[i] = weightInputs[i]
        for (i in dynamic.indices) all[userInputStart + i] = dynamic[i]
        @Suppress("UNCHECKED_CAST")
        val output = module.forward(*(all as Array<EValue>))
        require(output.isNotEmpty()) { "RVC runtime returned no output" }
        return output[0].toTensor().dataAsFloatArray
    }

    override fun close() {
        module.destroy()
    }
}
