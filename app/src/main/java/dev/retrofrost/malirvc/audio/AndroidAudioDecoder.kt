package dev.retrofrost.malirvc.audio

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder

/** Decodes any audio format supported by the device MediaCodec stack. */
object AndroidAudioDecoder {
    data class Decoded(val samples: FloatArray, val sampleRate: Int)

    fun decodeMono(context: Context, uri: Uri, maxSeconds: Int = 300): Decoded {
        val extractor = MediaExtractor()
        extractor.setDataSource(context, uri, null)
        try {
            var track = -1
            var inputFormat: MediaFormat? = null
            for (i in 0 until extractor.trackCount) {
                val f = extractor.getTrackFormat(i)
                val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
                if (mime.startsWith("audio/")) {
                    track = i
                    inputFormat = f
                    break
                }
            }
            require(track >= 0 && inputFormat != null) { "No audio track found" }
            extractor.selectTrack(track)
            val mime = inputFormat.getString(MediaFormat.KEY_MIME)!!
            val codec = MediaCodec.createDecoderByType(mime)
            codec.configure(inputFormat, null, null, 0)
            codec.start()
            try {
                val collected = FloatCollector()
                var currentFormat = inputFormat
                var channels = currentFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT, 1)
                var sampleRate = currentFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE, 16000)
                var pcmEncoding = AudioFormat.ENCODING_PCM_16BIT
                var inputEos = false
                var outputEos = false
                val info = MediaCodec.BufferInfo()

                while (!outputEos) {
                    if (!inputEos) {
                        val index = codec.dequeueInputBuffer(10_000)
                        if (index >= 0) {
                            val buffer = codec.getInputBuffer(index) ?: error("Decoder input buffer unavailable")
                            val size = extractor.readSampleData(buffer, 0)
                            if (size < 0) {
                                codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputEos = true
                            } else {
                                codec.queueInputBuffer(index, 0, size, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                        }
                    }

                    when (val index = codec.dequeueOutputBuffer(info, 10_000)) {
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            currentFormat = codec.outputFormat
                            channels = currentFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT, channels)
                            sampleRate = currentFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE, sampleRate)
                            if (currentFormat.containsKey(MediaFormat.KEY_PCM_ENCODING)) {
                                pcmEncoding = currentFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            }
                        }
                        MediaCodec.INFO_TRY_AGAIN_LATER -> Unit
                        else -> if (index >= 0) {
                            val buffer = codec.getOutputBuffer(index)
                            if (buffer != null && info.size > 0) {
                                buffer.position(info.offset)
                                buffer.limit(info.offset + info.size)
                                buffer.order(ByteOrder.nativeOrder())
                                when (pcmEncoding) {
                                    AudioFormat.ENCODING_PCM_FLOAT -> {
                                        while (buffer.remaining() >= 4 * channels) {
                                            var mono = 0f
                                            repeat(channels) { mono += buffer.float }
                                            collected.add(mono / channels)
                                        }
                                    }
                                    AudioFormat.ENCODING_PCM_8BIT -> {
                                        while (buffer.remaining() >= channels) {
                                            var mono = 0f
                                            repeat(channels) { mono += ((buffer.get().toInt() and 0xff) - 128) / 128f }
                                            collected.add(mono / channels)
                                        }
                                    }
                                    else -> { // PCM 16-bit is Android decoder default
                                        while (buffer.remaining() >= 2 * channels) {
                                            var mono = 0f
                                            repeat(channels) { mono += buffer.short / 32768f }
                                            collected.add(mono / channels)
                                        }
                                    }
                                }
                                require(collected.size <= sampleRate * maxSeconds) {
                                    "Input is longer than $maxSeconds seconds in this build"
                                }
                            }
                            outputEos = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            codec.releaseOutputBuffer(index, false)
                        }
                    }
                }
                return Decoded(collected.toArray(), sampleRate)
            } finally {
                codec.stop()
                codec.release()
            }
        } finally {
            extractor.release()
        }
    }

    private class FloatCollector {
        private var data = FloatArray(16_384)
        var size = 0
            private set
        fun add(v: Float) {
            if (size == data.size) data = data.copyOf(data.size * 2)
            data[size++] = v
        }
        fun toArray(): FloatArray = data.copyOf(size)
    }

    private fun MediaFormat.getInteger(key: String, fallback: Int): Int =
        if (containsKey(key)) getInteger(key) else fallback
}
