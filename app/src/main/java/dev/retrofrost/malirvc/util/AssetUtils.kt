package dev.retrofrost.malirvc.util

import android.content.Context
import java.io.File

object AssetUtils {
    fun exists(context: Context, path: String): Boolean = try {
        context.assets.open(path).close(); true
    } catch (_: Exception) { false }

    fun copyToFiles(context: Context, assetPath: String): File {
        val out = File(context.filesDir, "runtime/${assetPath.substringAfterLast('/')}")
        if (!out.exists() || out.length() == 0L) {
            out.parentFile?.mkdirs()
            context.assets.open(assetPath).use { input ->
                out.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return out
    }
}
