package dev.retrofrost.malirvc.model

object Half {
    fun toFloat(bits: Int): Float {
        val s = (bits ushr 15) and 0x1
        val e = (bits ushr 10) and 0x1f
        val f = bits and 0x3ff
        val out = when (e) {
            0 -> {
                if (f == 0) s shl 31
                else {
                    var frac = f
                    var exp = -14
                    while ((frac and 0x400) == 0) {
                        frac = frac shl 1
                        exp--
                    }
                    frac = frac and 0x3ff
                    (s shl 31) or ((exp + 127) shl 23) or (frac shl 13)
                }
            }
            31 -> (s shl 31) or 0x7f800000 or (f shl 13)
            else -> (s shl 31) or ((e - 15 + 127) shl 23) or (f shl 13)
        }
        return Float.fromBits(out)
    }
}
