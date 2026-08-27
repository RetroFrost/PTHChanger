package dev.retrofrost.malirvc

import android.content.Intent
import android.content.pm.PackageManager
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton
import com.google.android.material.card.MaterialCardView
import com.google.android.material.slider.Slider
import com.google.android.material.textfield.TextInputEditText
import dev.retrofrost.malirvc.inference.RvcPipeline
import dev.retrofrost.malirvc.model.PthCheckpoint
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var modelInfo: TextView
    private lateinit var audioInfo: TextView
    private lateinit var status: TextView
    private lateinit var progress: ProgressBar
    private lateinit var outputCard: MaterialCardView
    private lateinit var outputInfo: TextView
    private lateinit var speakerEdit: TextInputEditText
    private lateinit var pitchSlider: Slider
    private lateinit var pitchLabel: TextView

    private var modelFile: File? = null
    private var inputAudio: Uri? = null
    private var outputFile: File? = null
    private var player: MediaPlayer? = null
    private var speakerCount: Int = 1

    private val pickModel = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) importModel(uri)
    }
    private val pickAudio = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            inputAudio = uri
            audioInfo.text = displayName(uri)
        }
    }
    private val saveOutput = registerForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        val src = outputFile ?: return@registerForActivityResult
        if (uri != null) {
            lifecycleScope.launch(Dispatchers.IO) {
                contentResolver.openOutputStream(uri, "w")!!.use { out -> src.inputStream().use { it.copyTo(out) } }
                withContext(Dispatchers.Main) { status.text = "Saved ${displayName(uri)}" }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        bindViews()

        val hasVulkan = packageManager.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL)
        findViewById<TextView>(R.id.gpuStatus).text = if (hasVulkan) {
            "Vulkan available • RVC synthesizer targets Android GPU / Mali"
        } else {
            "Vulkan not reported by this device • GPU runtime may not start"
        }

        val missing = RvcPipeline.missingRuntimeAssets(this)
        if (missing.isNotEmpty()) {
            status.text = "Source build: runtime pack missing (${missing.joinToString { it.substringAfterLast('/') }})"
        }

        findViewById<MaterialButton>(R.id.selectModel).setOnClickListener {
            pickModel.launch(arrayOf("application/octet-stream", "application/zip", "*/*"))
        }
        findViewById<MaterialButton>(R.id.selectAudio).setOnClickListener {
            pickAudio.launch(arrayOf("audio/*"))
        }
        findViewById<MaterialButton>(R.id.convert).setOnClickListener { convert() }
        findViewById<MaterialButton>(R.id.playOutput).setOnClickListener { playOutput() }
        findViewById<MaterialButton>(R.id.saveOutput).setOnClickListener {
            outputFile?.let { saveOutput.launch(it.name) }
        }
        findViewById<MaterialButton>(R.id.shareOutput).setOnClickListener { shareOutput() }

        pitchSlider.addOnChangeListener { _, value, _ ->
            pitchLabel.text = "Pitch: ${value.toInt()} semitones"
        }
    }

    private fun bindViews() {
        modelInfo = findViewById(R.id.modelInfo)
        audioInfo = findViewById(R.id.audioInfo)
        status = findViewById(R.id.status)
        progress = findViewById(R.id.progress)
        outputCard = findViewById(R.id.outputCard)
        outputInfo = findViewById(R.id.outputInfo)
        speakerEdit = findViewById(R.id.speakerId)
        pitchSlider = findViewById(R.id.pitchSlider)
        pitchLabel = findViewById(R.id.pitchLabel)
    }

    private fun importModel(uri: Uri) {
        lifecycleScope.launch {
            setBusy(true, "Inspecting .pth checkpoint")
            try {
                val file = withContext(Dispatchers.IO) {
                    val dir = File(filesDir, "models").apply { mkdirs() }
                    val out = File(dir, "selected.pth")
                    contentResolver.openInputStream(uri)!!.use { input -> out.outputStream().use { input.copyTo(it) } }
                    out
                }
                val info = withContext(Dispatchers.IO) { PthCheckpoint(file).use { it.info } }
                modelFile = file
                speakerCount = info.speakerCount
                speakerEdit.setText(info.defaultSpeakerId.toString())
                modelInfo.text = buildString {
                    append(info.modelName).append(" • ").append(info.version)
                    append(" • ").append(info.sampleRate / 1000).append(" kHz")
                    append(" • F0 • ").append(info.tensors).append(" tensors")
                    append("\nContentVec ").append(info.embedder ?: "768")
                    append(" • ").append(info.vocoder ?: "HiFi-GAN")
                    append(" • speakers 0–").append(info.speakerCount - 1)
                    info.epoch?.let { append(" • epoch ").append(it) }
                }
                status.text = "Model ready"
            } catch (t: Throwable) {
                modelFile = null
                modelInfo.text = "Model rejected: ${t.message}"
                status.text = "Choose a compatible RVC v2 F0 .pth"
            } finally {
                setBusy(false)
            }
        }
    }

    private fun convert() {
        val model = modelFile ?: return showStatus("Select a .pth model first")
        val audio = inputAudio ?: return showStatus("Select an audio file first")
        val sid = speakerEdit.text?.toString()?.toIntOrNull() ?: 0
        if (sid !in 0 until speakerCount) return showStatus("Speaker ID must be 0–${speakerCount - 1}")
        val pitch = pitchSlider.value.toInt()

        lifecycleScope.launch {
            setBusy(true, "Starting conversion")
            outputCard.visibility = View.GONE
            try {
                val out = withContext(Dispatchers.IO) {
                    RvcPipeline(this@MainActivity).convert(model, audio, pitch, sid) { message ->
                        runOnUiThread { status.text = message }
                    }
                }
                outputFile = out
                outputInfo.text = "${out.name} • WAV • 40 kHz"
                outputCard.visibility = View.VISIBLE
                status.text = "Converted successfully"
            } catch (t: Throwable) {
                status.text = "Conversion failed: ${t.message ?: t.javaClass.simpleName}"
            } finally {
                setBusy(false)
            }
        }
    }

    private fun playOutput() {
        val file = outputFile ?: return
        player?.release()
        player = MediaPlayer().apply {
            setDataSource(file.absolutePath)
            prepare()
            start()
        }
    }

    private fun shareOutput() {
        val file = outputFile ?: return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "audio/wav"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Share converted audio"))
    }

    private fun displayName(uri: Uri): String {
        contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) return c.getString(0) ?: uri.lastPathSegment.orEmpty()
        }
        return uri.lastPathSegment ?: "audio"
    }

    private fun setBusy(busy: Boolean, message: String? = null) {
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        findViewById<MaterialButton>(R.id.convert).isEnabled = !busy
        findViewById<MaterialButton>(R.id.selectModel).isEnabled = !busy
        findViewById<MaterialButton>(R.id.selectAudio).isEnabled = !busy
        if (message != null) status.text = message
    }

    private fun showStatus(message: String) { status.text = message }

    override fun onDestroy() {
        player?.release()
        super.onDestroy()
    }
}
