package dev.retrofrost.malirvc.model

import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.LinkedHashMap
import java.util.zip.ZipFile

data class RvcCheckpointInfo(
    val modelName: String,
    val version: String,
    val sampleRate: Int,
    val f0: Boolean,
    val speakerCount: Int,
    val defaultSpeakerId: Int,
    val epoch: Int?,
    val step: Int?,
    val embedder: String?,
    val vocoder: String?,
    val tensors: Int
)

class PthCheckpoint(private val file: File) : Closeable {
    private val zip = ZipFile(file)
    private val prefix: String
    private val root: Map<*, *>

    val weights: LinkedHashMap<String, TorchTensorRef>
    val info: RvcCheckpointInfo

    init {
        val dataEntry = zip.entries().asSequence().firstOrNull { it.name.endsWith("/data.pkl") || it.name == "data.pkl" }
            ?: error("Not a supported PyTorch ZIP .pth: data.pkl is missing")
        prefix = dataEntry.name.removeSuffix("data.pkl")
        val parsed = zip.getInputStream(dataEntry).use { PthPickleReader(it.readBytes()).read() }
        root = parsed as? Map<*, *> ?: error("Checkpoint root is not a dictionary")

        @Suppress("UNCHECKED_CAST")
        val rawWeights = root["weight"] as? Map<Any?, Any?> ?: error("RVC checkpoint has no 'weight' dictionary")
        weights = LinkedHashMap()
        rawWeights.forEach { (k, v) ->
            if (k is String && v is TorchTensorRef) weights[k] = v
        }
        require(weights.isNotEmpty()) { "No tensor weights found" }

        val version = root["version"]?.toString() ?: "v1"
        val f0 = when (val v = root["f0"]) {
            is Boolean -> v
            is Number -> v.toInt() != 0
            else -> true
        }
        val config = root["config"] as? List<*>
        val sr = (root["sr"] as? Number)?.toInt()
            ?: (config?.lastOrNull() as? Number)?.toInt()
            ?: error("RVC sample rate missing")
        val emb = weights["emb_g.weight"] ?: error("Not an RVC synthesizer: emb_g.weight missing")
        val speakers = emb.shape.firstOrNull()?.toInt() ?: 1
        val defaultSid = (root["speakers_id"] as? Number)?.toInt()?.coerceIn(0, speakers - 1) ?: 0

        require(version == "v2") { "This build currently supports RVC v2 checkpoints; got $version" }
        require(f0) { "This build currently targets F0-enabled RVC models" }
        require(weights["enc_p.emb_phone.weight"]?.shape?.lastOrNull() == 768L) {
            "RVC v2 ContentVec dimension is not 768"
        }

        info = RvcCheckpointInfo(
            modelName = root["model_name"]?.toString() ?: file.nameWithoutExtension,
            version = version,
            sampleRate = sr,
            f0 = f0,
            speakerCount = speakers,
            defaultSpeakerId = defaultSid,
            epoch = (root["epoch"] as? Number)?.toInt(),
            step = (root["step"] as? Number)?.toInt(),
            embedder = root["embedder_model"]?.toString(),
            vocoder = root["vocoder"]?.toString(),
            tensors = weights.size
        )
    }

    fun readTensorAsFloat(name: String): FloatArray {
        val tensor = weights[name] ?: error("Weight '$name' is missing")
        require(tensor.isContiguous()) { "Non-contiguous checkpoint tensor is not supported: $name" }
        val storageEntry = zip.getEntry("${prefix}data/${tensor.storage.key}")
            ?: error("Storage ${tensor.storage.key} for '$name' is missing")
        val raw = zip.getInputStream(storageEntry).use { it.readBytes() }
        val count = tensor.elementCount
        require(count <= Int.MAX_VALUE) { "Tensor too large" }
        val out = FloatArray(count.toInt())
        val offsetBytes = tensor.storageOffset * tensor.storage.dtype.bytes
        require(offsetBytes >= 0 && offsetBytes + count * tensor.storage.dtype.bytes <= raw.size.toLong()) {
            "Tensor '$name' points outside its storage"
        }
        val bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN)
        bb.position(offsetBytes.toInt())
        when (tensor.storage.dtype) {
            TorchDType.FLOAT16 -> for (i in out.indices) out[i] = Half.toFloat(bb.short.toInt() and 0xffff)
            TorchDType.FLOAT32 -> for (i in out.indices) out[i] = bb.float
            else -> error("Weight '$name' has unsupported dtype ${tensor.storage.dtype}")
        }
        return out
    }

    override fun close() = zip.close()
}
