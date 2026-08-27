package dev.retrofrost.malirvc.model

enum class TorchDType(val bytes: Int) {
    FLOAT16(2), FLOAT32(4), INT64(8), INT32(4), UNKNOWN(1)
}

data class StorageRef(
    val key: String,
    val dtype: TorchDType,
    val elementCount: Long
)

data class TorchTensorRef(
    val storage: StorageRef,
    val storageOffset: Long,
    val shape: LongArray,
    val stride: LongArray
) {
    val elementCount: Long get() = shape.fold(1L) { a, b -> a * b }

    fun isContiguous(): Boolean {
        var expected = 1L
        for (i in shape.indices.reversed()) {
            if (shape[i] > 1 && stride[i] != expected) return false
            expected *= shape[i]
        }
        return true
    }
}
