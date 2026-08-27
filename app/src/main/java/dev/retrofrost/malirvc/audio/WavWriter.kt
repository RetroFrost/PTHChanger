package dev.retrofrost.malirvc.audio

import java.io.File
import java.io.FileOutputStream
import kotlin.math.roundToInt

object WavWriter {
    fun write16BitMono(file: File, samples: FloatArray, sampleRate: Int) {
        file.parentFile?.mkdirs()
        val dataSize = samples.size * 2
        FileOutputStream(file).use { out ->
            out.write("RIFF".toByteArray())
            writeLe32(out, 36 + dataSize)
            out.write("WAVEfmt ".toByteArray())
            writeLe32(out, 16)
            writeLe16(out, 1) // PCM
            writeLe16(out, 1) // mono
            writeLe32(out, sampleRate)
            writeLe32(out, sampleRate * 2)
            writeLe16(out, 2)
            writeLe16(out, 16)
            out.write("data".toByteArray())
            writeLe32(out, dataSize)
            for (sample in samples) {
                val s = (sample.coerceIn(-1f, 1f) * 32767f).roundToInt()
                writeLe16(out, s and 0xffff)
            }
        }
    }

    private fun writeLe16(out: FileOutputStream, value: Int) {
        out.write(value and 0xff)
        out.write((value ushr 8) and 0xff)
    }
    private fun writeLe32(out: FileOutputStream, value: Int) {
        out.write(value and 0xff)
        out.write((value ushr 8) and 0xff)
        out.write((value ushr 16) and 0xff)
        out.write((value ushr 24) and 0xff)
    }
}
