package dev.retrofrost.malirvc.audio

object LinearResampler {
    fun resample(input: FloatArray, fromHz: Int, toHz: Int): FloatArray {
        if (fromHz == toHz || input.isEmpty()) return input.copyOf()
        val outSize = ((input.size.toLong() * toHz) / fromHz).toInt().coerceAtLeast(1)
        val out = FloatArray(outSize)
        val scale = fromHz.toDouble() / toHz.toDouble()
        for (i in out.indices) {
            val pos = i * scale
            val a = pos.toInt().coerceIn(0, input.lastIndex)
            val b = (a + 1).coerceAtMost(input.lastIndex)
            val frac = (pos - a).toFloat()
            out[i] = input[a] * (1f - frac) + input[b] * frac
        }
        return out
    }
}
